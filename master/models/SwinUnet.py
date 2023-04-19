import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinWSSNet, SwinTransformerSys


class SwinUnet(nn.Module):
    def __init__(self, img_ch=15, output_ch=3):
        super(SwinUnet, self).__init__()

        self.swin_net = SwinWSSNet(img_size=48,
                                   patch_size=1,
                                   in_chans=img_ch,
                                   out_chans=output_ch,
                                   embed_dim=96,
                                   depths=[2, 2, 2],
                                   num_heads=[3, 6, 12],
                                   window_size=4,
                                   mlp_ratio=4.,
                                   qkv_bias=True,
                                   qk_scale=None,
                                   drop_rate=0.,
                                   drop_path_rate=0.1,
                                   ape=False,
                                   patch_norm=True,
                                   use_checkpoint=False)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits


class Standard_SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, output_ch=3, zero_head=False, vis=False):
        super(Standard_SwinUnet, self).__init__()
        self.output_ch = output_ch
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=img_size,
                                            patch_size=4,
                                            in_chans=15,
                                            num_classes=self.output_ch,
                                            embed_dim=96,
                                            depths=[2, 2, 6, 2],
                                            num_heads=[3, 6, 12, 24],
                                            window_size=7,
                                            mlp_ratio=4.,
                                            qkv_bias=True,
                                            qk_scale=None,
                                            drop_rate=0.0,
                                            drop_path_rate=0.1,
                                            ape=False,
                                            patch_norm=True,
                                            use_checkpoint=False)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits
