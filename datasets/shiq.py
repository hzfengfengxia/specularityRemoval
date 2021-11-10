import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

from collections import OrderedDict
import os
import cv2
import numpy as np
import random
from os.path import join

from torchvision import transforms
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class SHIQ(Dataset):
    def __init__(self,datadir,flip=False):
        super().__init__()
        self.datadir = os.path.normpath(datadir)
        self.flip = flip
        self.files = []

        fns = os.listdir(self.datadir)
        for fn in fns:
            if fn.endswith('_A.png'): self.files.append(fn[:-6])

        self._tensor = transforms.ToTensor()
        self.size = len(self.files)
        print('Load {} items in {} ...'.format(self.size,self.datadir))

    def __len__(self):
        return self.size

    def __getitem__(self,index):
        fn = self.files[index]
        # print(join(self.datadir,fn+'_A.png'))
        rgb = cv2.cvtColor(cv2.imread(join(self.datadir,fn+'_A.png'),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        diffuse = cv2.cvtColor(cv2.imread(join(self.datadir,fn+'_D.png'),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        label = cv2.imread(join(self.datadir,fn+'_T.png'),cv2.IMREAD_GRAYSCALE)
        if self.flip and random.random() < 0.5:
            rgb = cv2.flip(rgb, 0)
            diffuse = cv2.flip(diffuse, 0)
            label = cv2.flip(label, 0)
        if self.flip and random.random() < 0.5:
            rgb = cv2.flip(rgb, 1)
            diffuse = cv2.flip(diffuse, 1)
            label = cv2.flip(label, 1)

        rgb = self._tensor(rgb)
        diffuse = self._tensor(diffuse)
        label = self._tensor(label[...,np.newaxis])

        return {'rgb':rgb, 'diffuse':diffuse, 'label':label, 'fn':fn}
