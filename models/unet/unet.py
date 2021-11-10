""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import *

import numpy as np
from collections import OrderedDict

from .swin_transformer import SwinTransformer,StageModule
import math


class PatchedStage(nn.Module):
    def __init__(self, stage, downscaling_factor=2, window_size=8):
        super().__init__()
        self.stage = stage
        self.downscaling_factor = downscaling_factor
        self.window_size = window_size
    def forward(self,x):
        b,c,h,w = x.shape
        size = self.downscaling_factor*self.window_size
        dh = math.ceil(h/size)*size-h
        dw = math.ceil(h/size)*size-h
        x_pad = F.pad(x,[0,dw,0,dh],mode='reflect')
        y = self.stage(x_pad)
        return y[:,:,:h//self.downscaling_factor,:w//self.downscaling_factor]

class SwinUNetModel(nn.Module):
    def __init__(self, opt, in_channels=3, out_channels=3, bilinear=True, act=None):
        super().__init__()
        self.opt = opt
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.register_buffer('x_mean', torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None]))
        self.register_buffer('x_std', torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None]))

        self.inc = DoubleConv(in_channels, 64)
        stage = StageModule(in_channels=64, hidden_dimension=128, layers=2,
                downscaling_factor=2, num_heads=3, head_dim=32,
                window_size=8, relative_pos_embedding=True)
        self.down1 = PatchedStage(stage,downscaling_factor=2, window_size=8)

        stage = StageModule(in_channels=128, hidden_dimension=256, layers=2,
                downscaling_factor=2, num_heads=6, head_dim=32,
                window_size=8, relative_pos_embedding=True)
        self.down2 = PatchedStage(stage,downscaling_factor=2, window_size=8)

        stage = StageModule(in_channels=256, hidden_dimension=512, layers=6,
                downscaling_factor=2, num_heads=12, head_dim=32,
                window_size=8, relative_pos_embedding=True)
        self.down3 = PatchedStage(stage,downscaling_factor=2, window_size=8)

        factor = 2 if bilinear else 1
        stage = StageModule(in_channels=512, hidden_dimension=1024//factor, layers=2,
                downscaling_factor=2, num_heads=24, head_dim=32,
                window_size=8, relative_pos_embedding=True)
        self.down4 = PatchedStage(stage,downscaling_factor=2, window_size=8)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        assert act in [None, 'relu', 'sigmoid']
        if act is None:
            act = nn.Sequential()
        elif act == 'relu':
            act = nn.ReLU(inplace=True)
        elif act == 'sigmoid':
            act = nn.Sigmoid()
        self.outc = OutConv(64, out_channels, act=act)
        torch.cuda.empty_cache()

    def forward(self, x):
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.down4(x4)
        # print(x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetModel(nn.Module):
    def __init__(self, opt, in_channels=3, out_channels=3, bilinear=True, act=None):
        super().__init__()
        self.opt = opt
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.register_buffer('x_mean', torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None]))
        self.register_buffer('x_std', torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None]))

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        assert act in [None, 'relu', 'sigmoid']
        if act is None:
            act = nn.Sequential()
        elif act == 'relu':
            act = nn.ReLU(inplace=True)
        elif act == 'sigmoid':
            act = nn.Sigmoid()
        self.outc = OutConv(64, out_channels, act=act)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNet(nn.Module):
    def __init__(self, opt, in_channels=3, bilinear=True):
        super(UNet, self).__init__()
        self.opt = opt
        self.in_channels = in_channels
        self.bilinear = bilinear

        # self.register_buffer('x_mean', torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None]))
        # self.register_buffer('x_std', torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None]))

        self.detection = UNetModel(opt, in_channels, 1, bilinear, act='sigmoid')
        self.revomal = SwinUNetModel(opt, in_channels, 3, bilinear, act='sigmoid')

    def forward(self, _input, mode='test', **kwargs):
        
        # x = (_input['rgb'] - self.x_mean) / self.x_std
        x = _input['rgb']
        probability = self.detection(x)
        label = probability>0.5
        diffuse = self.revomal(x*(1-probability))
        
        mask = (probability<0.5).type(torch.float32)
        y = (diffuse+mask*x)/(1+mask)

        _output = OrderedDict()
        _output['coarse'] = diffuse
        _output['diffuse'] = y
        _output['probability'] = probability
        _output['label'] = label

        return _output
