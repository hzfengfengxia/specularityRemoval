import argparse
import os
import time
import math
class Options(object):
    def __init__(self):
        # create ArgumentParser() obj
        # formatter_class For customization to help document input formatter-class
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # call add_argument() to add parser
    def init(self):
        # add parser
        self.parser.add_argument('--id', type=str, default='UNet', help='the id to save and load model')
        self.parser.add_argument('--model', type=str, default='UNet', help='the model name')
        self.parser.add_argument('--mode', type=str, default='train', help='train | val | test')
        self.parser.add_argument('--gpus', type=str, default='0', help='gpus')

        self.parser.add_argument('--rand', default=False, help='')
        self.parser.add_argument('--batchsize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--threads', type=int, default=8, help='number of threads to load data')

        self.parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--ckpt_update', type=int, default=1, help='frequency to save model')
        self.parser.add_argument('--logs', type=str, default='./logs', help='training information are saved here')
        self.parser.add_argument('--load_epoch', type=int, default=-1, help='select checkpoint, default is the latest')
        self.parser.add_argument('--reload', '-r', action='store_true', help='resume from checkpoint')

        self.parser.add_argument('--optim', type=str, default='Adam', help='optimizer')
        self.parser.add_argument('--epochs', type=int, default=30, help='the all epochs')
        self.parser.add_argument('--reset_epoch', default=False, help='reset epoch')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--lr_decay', type=float, default=1.0, help='learning rate decay')
        self.parser.add_argument('--weight_decay', type=float, default=0, help='learning rate decay')
        self.parser.add_argument('--lr_update', type=int, default=0, help='learning rate update frequency')
        self.parser.add_argument('--lr_min', type=float, default=1e-5, help='min learning rate')

        self.parser.add_argument('--savedir', type=str, default='./results', help='dir to save the results')
        self.parser.add_argument('--saveimg', default=False, help='save images when testing/evaluating')
        
        
    def paser(self,log=True):
        self.init()
        self.opt = self.parser.parse_args()

        for k,v in vars(self.opt).items():
            if isinstance(v,str):
                if v.lower() == 'true':
                    setattr(self.opt,k,True)
                elif v.lower() == 'false':
                    setattr(self.opt,k,False)
                elif v.lower() == 'none':
                    setattr(self.opt,k,None)

        if self.opt.threads == 0:
            self.opt.threads = self.opt.batchsize

        assert not self.opt.gpus is None or "CUDA_VISIBLE_DEVICES" in os.environ
        if not self.opt.gpus is None: os.environ["CUDA_VISIBLE_DEVICES"] = self.opt.gpus

        torch_seed = int(math.modf(time.time())[0]*1e8) if self.opt.rand else 0
        numpy_seed = int(math.modf(time.time())[0]*1e8) if self.opt.rand else 0
        random_seed = int(math.modf(time.time())[0]*1e8) if self.opt.rand else 0
        
        msg = 'torch seed {}, numpy seed {}, random seed {}'.format(
                torch_seed,numpy_seed,random_seed)
        print(msg)
        import torch
        torch.manual_seed(torch_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()
        import numpy
        numpy.random.seed(numpy_seed)
        import random
        random.seed(random_seed)

        torch.cuda.empty_cache()

        assert self.opt.optim in ['Adam', 'SGD']

        for k,v in vars(self.opt).items():
            print('{}: {}'.format(k,v))

        return self.opt
