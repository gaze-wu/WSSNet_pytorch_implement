import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['WSSNet']


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class final_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(final_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class WSSNet(nn.Module):
    def __init__(self, img_ch=15, output_ch=3):
        """
        :param img_ch: 48*48*15
        :param output_ch: 48*48*3
        """
        super(WSSNet, self).__init__()

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.bottom4 = conv_block(ch_in=256, ch_out=512)
        self.Up5 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)
        self.Up6 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv6 = conv_block(ch_in=256, ch_out=128)
        self.Up7 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv7 = final_conv_block(ch_in=128, ch_out=64)
        self.Output_Conv = nn.Conv2d(64, output_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = F.max_pool2d(x1, 2)
        # x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = F.max_pool2d(x2, 2)
        # x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = F.max_pool2d(x3, 2)
        # x4 = self.Maxpool(x3)
        # neck path
        x4 = self.bottom4(x4)
        # decoding + concat path
        d5 = self.Up5(x4)
        d5 = torch.cat((x3, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d6 = self.Up6(d5)
        d6 = torch.cat((x2, d6), dim=1)
        d6 = self.Up_conv6(d6)
        d7 = self.Up7(d6)
        d7 = torch.cat((x1, d7), dim=1)
        d7 = self.Up_conv7(d7)
        output = self.Output_Conv(d7)
        return output
