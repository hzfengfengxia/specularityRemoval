from typing import Optional
from options.options import Options
opt = Options().paser()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.network import Network
from datasets.shiq import SHIQ
from models.losses import *

import os
from collections import OrderedDict
import numpy as np
import cv2
import glob

if __name__ == '__main__':

    losses = ['BCE','Content','Style','Coarse']
    weights = {'BCE':1.0,'Content':1.0,'Style':0.1,'Coarse':1.0}
    metrics =['L1','SSIM']

    funcs = OrderedDict({
        'BCE':BCE(),'Content':Content(),'Style':Style(),'Coarse':Content(branch='coarse'),
        'L1':L1(),'SSIM':SSIM()
    })

    model = Network(opt,funcs,weights)

    shiq_train = SHIQ('/file/',flip=True)
    shiq_train = DataLoader(shiq_train,opt.batchsize,num_workers=opt.batchsize,pin_memory=True,drop_last=True)
    shiq_test = SHIQ('/file/',flip=False)
    shiq_test = DataLoader(shiq_test,opt.batchsize,num_workers=opt.batchsize,pin_memory=True,drop_last=False)

    if opt.mode == 'train':
        while model.epoch < opt.epochs:

            model.opt.mode = 'train'
            model.train(shiq_train,losses)

            if model.epoch % opt.ckpt_update == 0:
                model.opt.mode = 'test'
                volumes_test = model.test(shiq_test,metrics,'./results/SHIQ/')
                model.opt.mode = 'train'
    else:
        volumes_test = model.test(shiq_test,metrics,'./results/SHIQ/')