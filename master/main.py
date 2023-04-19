import argparse
import os
import sys

import torch
import torch.nn as nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset

sys.path.append("..")

from models.WSSNet import WSSNet
from models.SwinUnet import SwinWSSNet
from models.SwinUnet import Standard_SwinUnet
import pytorch_ssim

import models
from wss_dataloder import WssDataset
from utils import progress_bar, adjust_learning_rate, mkdir_p, save_model, adjust_cosine_learning_rate

from torch.utils.tensorboard import SummaryWriter

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__"))

parser = argparse.ArgumentParser(description='WSSNet training(Pytorch)')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
parser.add_argument('--arch', default='SwinUnet', choices=model_names, type=str, help='choosing network')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--bs', default=16, type=int, help='batch size')
parser.add_argument('--es', default=100, type=int, help='epoch size')
parser.add_argument('--evaluate', action='store_true', help='Evaluate without training')
parser.add_argument('--lr_decay', default='cosine', type=str, help='Ways to Reduce Learning Rate: cosine or decay')
parser.add_argument('--opt', default='adam', type=str, help='Select the type of optimizer: adam or sgd')


# parser.add_argument('--gpu', default=None, type=int,help='GPU id to use.')

def train(net, train_loader, optimizer, loss_func, loss_func2, device):
    if device == 'cuda':
        net.cuda()
    net.train()
    train_loss = 0
    ssim_loss = 0
    epsilon = 1e-12
    for batch_idx, (input_layers, wss, mask) in enumerate(train_loader):
        if device == 'cuda':
            input_layers, wss, mask = input_layers.cuda(), wss.cuda(), mask.cuda()
        # print("the input (wss)data's dtype is ", wss.dtype)

        optimizer.zero_grad()
        outputs = net(input_layers)

        # apply mask and calculate the MAE
        outputs = outputs * mask
        wss = wss * mask
        loss = loss_func(outputs, wss)

        # F_wss = Fx^2 + Fy^2 + Fz^2 : (1,3,48,48) >>> (1,1,48,48)
        outputs = torch.norm(outputs, dim=1, keepdim=True)
        wss = torch.norm(wss, dim=1, keepdim=True)

        # Get the F_wss of space by sqrt, and add epsilon to avoid the NAN
        outputs = torch.sqrt(torch.sum(outputs**2,dim=1,keepdim=True) + epsilon).cpu()
        wss = torch.sqrt(torch.sum(wss ** 2, dim=1, keepdim=True) + epsilon).cpu()

        # print("the input (wss)data's dtype is ", wss.dtype, "the input (out)data's dtype is ",outputs.dtype)
        loss2 = loss_func2(outputs, wss)

        total_loss = loss + 1.5 * loss2
        total_loss.backward()

        optimizer.step()

        train_loss += loss.item()
        ssim_loss += loss2.item()
        progress_bar(batch_idx, len(train_loader), 'L1_loss: %.8f ' % (train_loss / (batch_idx + 1)))

    return train_loss / (batch_idx + 1), ssim_loss / (batch_idx + 1)


def test(net, test_loader, loss_func, loss_func2, device):
    net.eval()
    test_loss = 0
    ssim_loss = 0
    epsilon = 1e-12
    with torch.no_grad():
        for batch_idx, (inputs, wss, mask) in enumerate(test_loader):
            inputs, wss, mask = inputs.to(device), wss.to(device), mask.to(device)
            outputs = net(inputs)

            # apply mask and calculate the MAE
            outputs = outputs * mask
            wss = wss * mask
            loss = loss_func(outputs, wss)

            # F_wss = Fx^2 + Fy^2 + Fz^2 : (1,3,48,48) >>> (1,1,48,48)
            outputs = torch.norm(outputs, dim=1, keepdim=True)
            wss = torch.norm(wss, dim=1, keepdim=True)

            # Get the F_wss of space by sqrt, and add epsilon to avoid the NAN
            outputs = torch.sqrt(torch.sum(outputs ** 2, dim=1, keepdim=True) + epsilon).cpu()
            wss = torch.sqrt(torch.sum(wss ** 2, dim=1, keepdim=True) + epsilon).cpu()
            loss2 = loss_func2(outputs, wss)

            test_loss += loss.item()
            ssim_loss += loss2.item()

            progress_bar(batch_idx, len(test_loader), 'L1_loss:%.8f' % (test_loss / (batch_idx + 1)))

    return test_loss / (batch_idx + 1), ssim_loss / (batch_idx + 1)


def main():
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # pretrain model
    #ckpt_dir = './pretrain_weight/swin_tiny_patch4_window7_224.pth'
    #ckpt = torch.load(ckpt_dir)

    print('the Network architecture is ', args.arch)

    # checkpoint
    args.checkpoint = './checkpoints/WSS/' + args.arch
    # checkpoint = './checkpoints/WSS/'
    # print('the path of checkpoint is ', checkpoint)
    print('the path of checkpoint is ', args.checkpoint)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data_loader
    train_data = WssDataset('../dataset/train.csv', '../dataset/train', transform=True)
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=4)
    val_data = WssDataset('../dataset/val.csv', '../dataset/val', transform=True)
    # val_loader = DataLoader(val_data, batch_size=args.bs, shuffle=True, num_workers=4)
    test_data = WssDataset('../dataset/test.csv', '../dataset/test', transform=True)
    # test_loader = DataLoader(val_data, batch_size=args.bs, shuffle=True, num_workers=4)

    # merge val_data and test data
    concat_dataset = ConcatDataset([val_data, test_data])
    val_loader = DataLoader(dataset=concat_dataset, batch_size=args.bs, shuffle=True, num_workers=4)

    # build the model
    print('>>>>>Building model..')
    #net = SwinWSSNet().to(device)
    net = WSSNet().to(device)

    # 多卡训练
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        if os.path.isfile(args.resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # loss func of MAE and SSIM
    Loss_func = nn.L1Loss()
    Loss_func2 = pytorch_ssim.SSIM()
    # optimizer

    if args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    elif args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # turn on the tensorboard
    writer = SummaryWriter()

    # train and val
    best_val_loss = 0xffff
    if not args.evaluate:
        for epoch in range(start_epoch, args.es):
            print('\nEpoch: %d   Learning rate: %f' % (epoch + 1, optimizer.param_groups[0]['lr']))

            # Lr scheduler
            if args.lr_decay == 'cosine':
                adjust_cosine_learning_rate(optimizer, epoch, lr_max=5e-4, lr_min=5e-7, reset_epoch=10)
            elif args.lr_decay == 'decay':
                adjust_learning_rate(optimizer, epoch, args.lr)

            # train
            train_loss, ssim_loss = train(net, train_loader, optimizer, Loss_func, Loss_func2, device)
            writer.add_scalar(tag="train/loss", scalar_value=train_loss, global_step=epoch)
            writer.add_scalar(tag="train/ssim_loss", scalar_value=ssim_loss, global_step=epoch)

            # test
            val_loss, val_ssim_loss = test(net, val_loader, Loss_func, Loss_func2, device)
            writer.add_scalar(tag="val/loss", scalar_value=val_loss, global_step=epoch)
            writer.add_scalar(tag="val/ssim_loss", scalar_value=val_ssim_loss, global_step=epoch)

            # save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(net, optimizer, epoch,
                           os.path.join(args.checkpoint, 'best_model_for_val_{}epoch.pth'.format(epoch)))
            # save the model per epoch
            if epoch % 5 == 0 and epoch != 0:
                save_model(net, optimizer, epoch,
                           os.path.join(args.checkpoint, 'model_{}epoch.pth'.format(epoch)))

        # close the tensorboard writer
        writer.close()
        # save the final model
        save_model(net, optimizer, args.es, os.path.join(args.checkpoint, 'last_model.pth'))


if __name__ == '__main__':
    main()
