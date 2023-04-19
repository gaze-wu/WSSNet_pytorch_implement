import os
import h5py
import models
import torch

from utils import h5util
from PatchInputHandler import PatchInputHandler


def predict_all_rows(dataset_file, distances, net, device, output_dir, output_filename):
    """
        Run predictions per row from an HDF5 file
    """
    scale_dist = 100
    # prepare input
    with h5py.File(dataset_file, mode='r') as hdf5:
        len_indexes = len(hdf5['wss_vector'])
        wall_coords = hdf5.get('xyz0')[0]

    pc = PatchInputHandler(dataset_file, scale_dist, 48, distances, False)
    for i in range(len_indexes):
        print(f"Processing row {i}/{len_indexes}")
        data_pairs = pc.load_patches_from_index_file(i)  # 5 * (2,48,48,3)
        wss_true = data_pairs['wss']

        # Due to the differences in data formats between tf and pytorch
        # the dimensions of the data should be transformed: (2,48,48,3) >>> (2,3,48,48)
        data_pairs['xyz0'] = data_pairs['xyz0'].reshape(2, 3, 48, 48)
        data_pairs['xyz1'] = data_pairs['xyz1'].reshape(2, 3, 48, 48)
        data_pairs['xyz2'] = data_pairs['xyz2'].reshape(2, 3, 48, 48)
        data_pairs['v1'] = data_pairs['v1'].reshape(2, 3, 48, 48)
        data_pairs['v2'] = data_pairs['v2'].reshape(2, 3, 48, 48)
        input_data = [torch.tensor(data_pairs['xyz0'], dtype=torch.float32),
                      torch.tensor(data_pairs['xyz1'], dtype=torch.float32),
                      torch.tensor(data_pairs['xyz2'], dtype=torch.float32),
                      torch.tensor(data_pairs['v1'], dtype=torch.float32),
                      torch.tensor(data_pairs['v2'], dtype=torch.float32)]
        input_data = torch.cat(input_data, dim=1)
        input_data = input_data.to(device)

        # inference
        net.eval()
        with torch.no_grad():
            wss_pred = net(input_data)

        wss_pred = wss_pred.cpu().numpy()
        wss_pred = wss_pred.reshape(2, 48, 48, 3)
        wss_pred = pc.unpatchify(wss_pred)

        h5util.save_predictions(output_dir, output_filename, f"wss", wss_pred, compression='gzip', auto_expand=True)
        h5util.save_predictions(output_dir, output_filename, f"wss_true", wss_true, compression='gzip',
                                auto_expand=True)

    # only save the wall coordinates once
    h5util.save_predictions(output_dir, output_filename, f"xyz0", wall_coords, compression='gzip', auto_expand=True)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # your pre-trained ckpt
    model = "best_model_for_val_187epoch.pth"
    model_patch = f"checkpoints/WSS/SwinUnet/{model}"

    # Put all your hdf5 input files here
    input_dir = "examples"
    # Predictions will be saved here
    output_dir = "examples"

    # input_filename = "ch11_sheet.h5"
    # input_filename = "ch11_sheet_noise.h5"
    # input_filename = "ch11_clean.h5"

    # input_filename = "ch61_clean.h5"
    # input_filename = "ch61_sheet.h5"
    # input_filename = "ch61_sheet_noise.h5"

    #input_filename = "ch64_clean.h5"
    #input_filename = "ch64_sheet.h5"
    #input_filename = "ch64_sheet_noise.h5"

    #input_filename = "ch70_clean.h5"
    #input_filename = "ch70_sheet.h5"
    #input_filename = "ch70_sheet_noise.h5"

    #input_filename = "ch46_clean.h5"
    #input_filename = "ch46_sheet.h5"
    #input_filename = "ch46_sheet_noise.h5"

    #input_filename = "ch39_clean.h5"
    #input_filename = "ch39_sheet.h5"
    input_filename = "ch39_sheet_noise.h5"



    output_filename = "prediction_{}_{}_.h5".format(model, input_filename)

    # Put your model here
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Presets, modified if necessary
    input_shape = (48, 48)
    distances = [1.0, 2.0]

    print('>>>>>Building model..')
    net = models.SwinWSSNet().to(device)
    # load the weight
    checkpoint = torch.load(model_patch, map_location=lambda storage, loc: storage)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}  # 去掉模型结构中的'module.'前缀
    net.load_state_dict(state_dict)
    print("the weight has been load.")

    dataset_file = f"{input_dir}/{input_filename}"

    if not os.path.exists(dataset_file):
        print(f"Processing file dose not exists: {input_filename}")
    else:
        print(f"Processing case {input_filename}")
        print(distances)
        predict_all_rows(dataset_file, distances, net, device, output_dir, output_filename)

    print("the case is :", input_filename, "the model is: ", model)
    print("the output is :", output_filename)


if __name__ == '__main__':
    main()
