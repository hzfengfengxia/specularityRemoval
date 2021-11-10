import sys
import numpy as np
import torch
import cv2

def tensor2img(image_tensor):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor.cpu().float().numpy()
    if len(image_numpy.shape) == 3:
        assert image_numpy.shape[0] in [1,3]
        if image_numpy.shape[0] == 3:
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
        else:
            image_numpy = image_numpy[0]
    image_numpy = np.uint8(np.clip(image_numpy*255.0, 0, 255))
    if len(image_numpy.shape) == 3:
        image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    return image_numpy

    #return np.uint8(np.clip(image_numpy*255.0, 0, 255))
