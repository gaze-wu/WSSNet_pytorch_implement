import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from random import randrange, randint
from utils import augmentation as aug


def augment_roll(shift, xyz0, xyz1, xyz2, v1, v2, wss, mask):
    # shift roll
    xyz0 = np.roll(xyz0, shift, axis=0)
    xyz1 = np.roll(xyz1, shift, axis=0)
    xyz2 = np.roll(xyz2, shift, axis=0)

    v1 = np.roll(v1, shift, axis=0)
    v2 = np.roll(v2, shift, axis=0)

    wss = np.roll(wss, shift, axis=0)
    mask = np.roll(mask, shift, axis=0)
    return xyz0, xyz1, xyz2, v1, v2, wss, mask


def augment_noise(v1, v2):
    rnd = randint(0, 1)
    # 50% chance to add noise
    if rnd > 0:
        # noise between 1-4% venc
        noiseLevel = randint(1, 4)
        maxStd = 1.5 * noiseLevel / 100.

        # print('ADDING noise', maxStd)
        # we assume the same noise on every sheet
        # although different on each component
        noise = np.random.normal(0, maxStd, v1.shape)
        for c in range(0, 3):
            noise[..., c] = cv2.GaussianBlur(noise[..., c], (3, 3), 0)

        v1 += noise
        v2 += noise

    return v1, v2


def augment_rotation(xyz0, xyz1, xyz2, v1, v2, wss):
    rnd = randint(0, 4)
    if rnd == 0:
        # print('no rotation')
        return xyz0, xyz1, xyz2, v1, v2, wss
    else:
        randAngle = randrange(0, 360)
        randAxis = randint(0, 2)
        # perform augmentation
        # TODO: separate rotation matrix
        xyz0 = aug.rotate(xyz0, randAngle, axis=randAxis)
        xyz1 = aug.rotate(xyz1, randAngle, axis=randAxis)
        xyz2 = aug.rotate(xyz2, randAngle, axis=randAxis)
        v1 = aug.rotate(v1, randAngle, axis=randAxis)
        v2 = aug.rotate(v2, randAngle, axis=randAxis)
        wss = aug.rotate(wss, randAngle, axis=randAxis)

    return xyz0, xyz1, xyz2, v1, v2, wss


def load_sheet_data(hd5path, row_idx, dist1, dist2):
    with h5py.File(hd5path, 'r') as hl:
        xyz0 = hl.get('xyz0')[0]

        xyz1 = hl.get(f'xyz{dist1}')[0]
        xyz2 = hl.get(f'xyz{dist2}')[0]

        v1 = hl.get(f'v{dist1}')[row_idx]
        v2 = hl.get(f'v{dist2}')[row_idx]

        wss = hl.get('wss_vector')[row_idx]
        # print('the file is',hd5path,' the row idx is', row_idx, ' the dist1 and the dist2 is ', dist1, dist2)
        mask = hl.get('wss_mask')[0]

        return xyz0.astype('float32'), xyz1.astype('float32'), xyz2.astype('float32'), \
               v1.astype('float32'), v2.astype('float32'), \
               wss.astype('float32'), mask.astype('float32')


# xyz0, xyz1, xyz2, v1, v2, wss, mask = load_sheet_data('dataset/train/ch01_clean.h5',0,0.3,0.4)

def get_patch(img, patch_start, patch_size):
    return img[:, patch_start:patch_start + patch_size]


def get_patches(xyz0, xyz1, xyz2, v1, v2, wss, mask, randomize):
    patch_size = xyz0.shape[0]  # 48
    if randomize:
        # start from col 3 instead of 0, because of high WSS
        patch_start = randrange(3, xyz0.shape[1] - patch_size)
    else:
        patch_start = 3
    # get patch
    xyz0 = get_patch(xyz0, patch_start, patch_size)
    xyz1 = get_patch(xyz1, patch_start, patch_size)
    xyz2 = get_patch(xyz2, patch_start, patch_size)
    v1 = get_patch(v1, patch_start, patch_size)
    v2 = get_patch(v2, patch_start, patch_size)
    wss = get_patch(wss, patch_start, patch_size)
    mask = get_patch(mask, patch_start, patch_size)

    return xyz0, xyz1, xyz2, v1, v2, wss, mask


def load_patches_from_index_file(data_dir, csv_row, transform):
    file_name = '{}/{}'.format(data_dir, csv_row[0])
    idx = int(csv_row[1])
    dist1 = float(csv_row[2])
    dist2 = float(csv_row[3])

    xyz0, xyz1, xyz2, v1, v2, wss, mask = load_sheet_data(file_name, idx, dist1, dist2)
    xyz0, xyz1, xyz2, v1, v2, wss, mask = get_patches(xyz0, xyz1, xyz2, v1, v2, wss, mask, transform)

    # prepare to Aug
    if transform:
        # Augmentation - Rotate
        xyz0, xyz1, xyz2, v1, v2, wss = augment_rotation(xyz0, xyz1, xyz2, v1, v2, wss)

        # Augmentation - Noise
        v1, v2 = augment_noise(v1, v2)
    # Augmentation - Roll
    shift = randint(-5, 5)

    xyz0, xyz1, xyz2, v1, v2, wss, mask = augment_roll(shift, xyz0, xyz1, xyz2, v1, v2, wss, mask)
    # Augmentation - Translate ref point
    # pick a random point in xyz0
    pos_x = randrange(0, xyz0.shape[0])
    pos_y = randrange(0, xyz0.shape[1])

    # get the reference point 0 on the wall
    # make sure to actually keep the value using .copy()
    ref_coord = xyz0[pos_x, pos_y, :].copy()

    # recenter the coordinates to the ref point
    xyz0 -= ref_coord
    xyz1 -= ref_coord
    xyz2 -= ref_coord

    # rescale the data so it fits with velocity range
    xyz0 /= 100
    xyz1 /= 100
    xyz2 /= 100
    #
    # xyz0 = xyz0.reshape(3, 48)

    xyz0 = xyz0.reshape(3, 48, 48)
    xyz1 = xyz1.reshape(3, 48, 48)
    xyz2 = xyz2.reshape(3, 48, 48)
    v1 = v1.reshape(3, 48, 48)
    v2 = v2.reshape(3, 48, 48)
    wss = wss.reshape(3, 48, 48)
    # print(mask.shape)
    mask = mask.reshape(1,48,48)

    output_list = [torch.tensor(xyz0, dtype=torch.float32), torch.tensor(xyz1, dtype=torch.float32), torch.tensor(xyz2, dtype=torch.float32),
                   torch.tensor(v1, dtype=torch.float32), torch.tensor(v2, dtype=torch.float32)]
    output = torch.cat(output_list, dim=0)  # 1*15*48*48
    wss = torch.tensor(wss, dtype=torch.float32)
    return output, wss, mask


class WssDataset(Dataset):
    def __init__(self, csv_file, h5_dir, transform=None):
        # read the csv file, h5file's dir and trans whether or not
        self.csv_file = pd.read_csv(csv_file)
        self.h5_dir = h5_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        row_data = self.csv_file.iloc[idx]
        # print('the idx is ',idx)
        # print(row_data)
        # print('\n')
        input_layers, wss, mask = load_patches_from_index_file(self.h5_dir, row_data, self.transform)
        # print(input_layers.shape)
        # 48*48*15 ~ 48*48*3
        return input_layers, wss, mask
