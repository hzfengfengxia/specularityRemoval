import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import math

from .ssim import SSIM

class L1(nn.Module):
    def __init__(self,branch='diffuse',target='diffuse'):
        super().__init__()
        self.branch = branch
        self.target = target

    def forward(self,_output,_input):
        if not self.branch in _output or not self.target in _input:
            return torch.tensor(0).to(_input[self.targe].device)
        else:
            return F.l1_loss(_output[self.branch],_input[self.target])

class BCE(nn.Module):
    def __init__(self,branch='probability',target='label'):
        super().__init__()
        self.branch = branch
        self.target = target
    def forward(self,_output,_input):
        if not self.branch in _output or not self.target in _input:
            return torch.tensor(0).to(_input[self.targe].device)
        else:
            return F.binary_cross_entropy(_output[self.branch],_input[self.target])

class Content(nn.Module):
    def __init__(self,branch='diffuse',target='diffuse'):
        super().__init__()
        self.branch = branch
        self.target = target

    def compute_gradient(self,img):
        gradx = img[...,1:,:]-img[...,:-1,:]
        grady = img[...,1:]-img[...,:-1]
        return gradx,grady

    def forward(self,_output,_input):
        if not self.branch in _output or not self.target in _input:
            return torch.tensor(0).to(_input[self.targe].device)
        else:
            y = _output[self.branch]
            t = _input[self.target]
            mse = F.mse_loss(y,t)
            y_gradx, y_grady = self.compute_gradient(y)
            t_gradx, t_grady = self.compute_gradient(t)
            grad = F.l1_loss(y_gradx,t_gradx)+F.l1_loss(y_grady,t_grady)
            return mse*0.2+grad*0.4

class Style(nn.Module):
    def __init__(self,branch='diffuse',target='diffuse'):
        super().__init__()
        self.branch = branch
        self.target = target

        self.register_buffer('mean', torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None]))
        self.register_buffer('std', torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None]))

        self.vgg = models.vgg19(pretrained=True).features.cuda()
        # print(self.vgg)
        for par in self.vgg.parameters():
            par.requires_grad = False
        self.style_layers = [0, 5, 10, 19, 28]

    def gram_matrix(self,input):
        batch = input.size()[0]
        channels = input.size()[1]
        height = input.size()[2]
        width = input.size()[3]
        input = input.clone().view(batch * channels, height * width)
        G = torch.mm(input, input.t())
        G = G.div(batch * channels * height * width)
        return G

    def forward(self,_output,_input):
        if not self.branch in _output or not self.target in _input:
            return torch.tensor(0).to(_input[self.target].device)
        y = torch.clamp(_output[self.branch].clone(),0,1)
        t = torch.clamp(_input[self.target].clone(),0,1)
        feat_y = (y - self.mean) / self.std
        feat_t = (t - self.mean) / self.std
        
        loss = 0
        num_layer = 0
        for l in range(len(self.vgg)):
            feat_y = self.vgg[l](feat_y)
            feat_t = self.vgg[l](feat_t)
            if l in self.style_layers:
                loss += F.mse_loss(self.gram_matrix(feat_y),self.gram_matrix(feat_t))*1e4
        return loss
