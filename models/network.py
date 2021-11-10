import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import os
import shutil
from os.path import normpath,join,basename
from collections import OrderedDict
import numpy as np
import cv2

import time
from tqdm import tqdm,trange
import sys
sys.path.append('.')

from .utils import tensor2img
from .unet.unet import UNet

class Core(nn.Module):
    def __init__(self,opt,loss_func=None,loss_weight=None):
        super().__init__()

        ### initial
        self.opt = opt
        self.net = eval(self.opt.model)(opt).cuda()
        self.loss_func = nn.ModuleDict(loss_func).cuda() if not loss_func is None else {}
        self.loss_weight = loss_weight if not loss_weight is None else {}

        total_num = sum(p.numel() for p in self.net.parameters())
        trainable_num = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('Total pars: {:.3f}M, Trainable pars: {:.3f}M'.format(total_num/1e6,trainable_num/1e6))

    def forward(self,_input,mode,loss_item,**kwargs):
        loss_dict = OrderedDict()
        if mode == 'train':
            _output = self.net(_input,mode=mode,**kwargs)
            for key in loss_item:
                loss_dict[key] = self.loss_func[key](_output,_input)
        else:
            with torch.no_grad():
                _output = self.net(_input,mode=mode,**kwargs)
                for key in loss_item:
                    loss_dict[key] = self.loss_func[key](_output,_input)
        return _output, loss_dict

class Network(object):
    def __init__(self,opt,loss_func=None,loss_weight=None):
        super().__init__()

        ### initial
        self.opt = opt
        self.core = Core(opt,loss_func,loss_weight)
        print(self.core.net)

        self.loss_weight = loss_weight
        self._input = OrderedDict()
        self._output = OrderedDict()

        #### reload
        self.epoch = 0
        self.iter = 0
        self.logs = os.path.join(opt.logs,opt.id)
        if not opt.reload and os.path.exists(self.logs):
            shutil.rmtree(self.logs)
        os.makedirs(self.logs,exist_ok=True)

        self.checkpoints = os.path.join(opt.checkpoints,opt.id)
        os.makedirs(self.checkpoints,exist_ok=True)

        ckpt = os.path.join(self.checkpoints,self.opt.model.lower()+"-epoch-{:04d}.ckpt".format(opt.load_epoch))\
                if opt.load_epoch >= 0 else os.path.join(self.checkpoints,self.opt.model.lower()+"-latest.pt")
        if (self.opt.reload or not self.opt.mode == 'train'):
            self.load_ckpt(ckpt)
        if self.opt.reset_epoch:
            self.epoch = 0

        ### initial for training
        if self.opt.mode == 'train':
            if self.opt.optim == 'Adam':
                self.optimizer = torch.optim.Adam(self.core.parameters(),lr=self.opt.lr,weight_decay=self.opt.weight_decay,betas=(0.9, 0.999))
            else:
                self.optimizer = torch.optim.SGD(self.core.parameters(),lr=self.opt.lr,weight_decay=self.opt.weight_decay,momentum=0.9)

        if self.opt.mode == 'train':
            os.makedirs(self.logs,exist_ok=True)
            self.writer = SummaryWriter(self.logs)
        else:
            self.writer = None
        print("Use CUDNN Benchmark: ",torch.backends.cudnn.benchmark)
        print("Use Deterministic Algorithm: ",torch.backends.cudnn.deterministic)

        torch.cuda.empty_cache()

    def save_ckpt(self,ckpt):
        state_dict = {
            'model': self.core.net.state_dict(),
            'epoch': self.epoch,
            'iter': self.iter,
        }
        torch.save(state_dict,ckpt)

    def load_ckpt(self,ckpt):
        print('Load from {} ... '.format(ckpt))
        if os.path.exists(ckpt):
            state_dict = torch.load(ckpt,map_location='cpu')
            self.core.net.load_state_dict(state_dict['model'],strict=True)
            self.epoch = state_dict['epoch']
            self.iter = state_dict['iter']
            del state_dict
            torch.cuda.empty_cache()
            os.makedirs(self.logs,exist_ok=True)
            print('Load successfully!')
        else:
            print('No checkpoint found!')

    def set_input(self,_input):
        self._input = OrderedDict()
        for key in _input:
            self._input[key] = _input[key].cuda() if torch.is_tensor(_input[key]) and not _input[key].is_cuda else _input[key]

    def set_output(self,_output):
        self._output = OrderedDict()
        for key in _output.keys():
            if torch.is_tensor(_output[key]):
                self._output[key] = _output[key].detach()

    def set_parameters(self,**kwargs):
            self.core.net.set_parameters(**kwargs)

    def save(self,keys,savedir):
        fns = self._input['fn']
        B = self._input['rgb'].shape[0]
        os.makedirs(savedir,exist_ok=True)
        for key in keys:
            if key in self._input:
                for b in range(B):
                    item = tensor2img(self._input[key][b])
                    cv2.imwrite(join(savedir,fns[b]+'_{}_input.png'.format(key)),item)
            if key in self._output:
                for b in range(B):
                    item = tensor2img(self._output[key][b])
                    cv2.imwrite(join(savedir,fns[b]+'_{}_output_{}.png'.format(key,self.opt.id)),item)

    def train(self,dataloader,losses,**kwargs):
        # loss_dict = OrderedDict()
        torch.cuda.empty_cache()
        assert self.opt.mode == 'train'
        self.core.train()
        if self.opt.lr_update > 0:
            self.opt.lr = max(self.opt.lr_min,self.opt.lr*self.opt.lr_decay**(self.epoch//self.opt.lr_update))
        else:
            self.opt.lr = self.opt.lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.opt.lr

        volumes = OrderedDict({'Size':0})
        for key in losses:
            volumes[key] = 0

        dataloader = tqdm(dataloader,dynamic_ncols=True)
        for idx, _input in enumerate(dataloader):
            self.set_input(_input)
            self.optimizer.zero_grad()
            _output, loss_dict = self.core(self._input,self.opt.mode,losses,**kwargs)
            self.set_output(_output)

            loss = 0
            msg = 'Epoch {}/{} |'.format(self.epoch+1,self.opt.epochs)
            volumes['Size'] += _input['rgb'].shape[0]
            for key in loss_dict:
                loss = loss + loss_dict[key]*self.loss_weight[key]
                value = loss_dict[key].detach().item()
                volumes[key] += value*_input['rgb'].shape[0]
                msg += '' if value == 0 else ' {}:{:.4f} |'.format(key,value)
                self.writer.add_scalar("TRAIN/"+key,loss_dict[key].detach().item(),self.iter)
            loss.backward()
            self.optimizer.step()
            self.writer.flush()
            
            msg += ' Overall ::'
            for key in loss_dict:
                msg += '' if volumes[key] == 0 else ' {}:{:.4f} |'.format(key,volumes[key]/volumes['Size'])
                self.writer.add_scalar("TRAIN/Overall_"+key,volumes[key]/volumes['Size'],self.iter)
            dataloader.set_postfix_str(msg)
            self.iter += 1
        self.epoch += 1

        self.save_ckpt(os.path.join(self.checkpoints,self.opt.model.lower()+"-latest.pt"))
        if self.epoch % self.opt.ckpt_update == 0:
            self.save_ckpt(os.path.join(self.checkpoints,"{}-epoch-{:04}.ckpt".format(self.opt.model.lower(),self.epoch)))

        torch.cuda.empty_cache()
    
    def test(self,dataloader,metrics,savedir,**kwargs):
        assert self.opt.mode in ['val','test']
        self.core.eval()
        # torch.cuda.empty_cache()
        volumes = OrderedDict({'Size':0})
        for key in metrics:
            volumes[key] = 0

        with torch.no_grad():
            dataloader = tqdm(dataloader,dynamic_ncols=True)
            for idx, _input in enumerate(dataloader):
                self.set_input(_input)
                _output, loss_dict = self.core(self._input,self.opt.mode,metrics,**kwargs)
                self.set_output(_output)
                volumes['Size'] += _input['rgb'].shape[0]
                for key in metrics:
                    volumes[key] += loss_dict[key].item()*_input['rgb'].shape[0]
                msg = 'Epoch {}/{} | Overall :: '.format(self.epoch,self.opt.epochs)
                for key in loss_dict:
                    msg += ' {}:{:.4f} |'.format(key,volumes[key]/volumes['Size'])
                dataloader.set_postfix_str(msg)

                if self.opt.saveimg:
                    self.save(['rgb','diffuse','label'],savedir)

            if not self.writer is None:
                for key in metrics:
                    self.writer.add_scalar(self.opt.mode.upper()+"/Mean "+key,volumes[key]/volumes['Size'],self.epoch)
                self.writer.flush()
                 