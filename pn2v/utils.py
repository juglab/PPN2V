############################################
#   Utility Functions
############################################

import torch.optim as optim
import os
import torch
import torch.nn as nn
import sys
import torchvision
import numpy as np

def printNow(string,a="",b="",c="",d="",e="",f=""):
    print(string,a,b,c,d,e,f)
    sys.stdout.flush()

def imgToTensor(img):
    '''
    Convert a 2D single channel image to a pytorch tensor.
    '''
    img.shape=(img.shape[0],img.shape[1],1)
    imgOut = torchvision.transforms.functional.to_tensor(img.astype(np.float32))
    return imgOut
    
def PSNR(gt, pred, range_=255.0 ):
    mse = np.mean((gt - pred)**2)
    return 20 * np.log10((range_)/np.sqrt(mse))

def normalize(img, mean, std):
    """Normalizes image `img` with mean `mean` and standard deviation `std`

        Parameters
        ----------
        img: array
             image
        mean: float
            mean
        std: float
            standard deviation
    """
    zero_mean = img - mean
    return zero_mean/std

def denormalize(x, mean, std):
    """Denormalizes `x` with mean `mean` and standard deviation `std`

        Parameters
        ----------
        x: array
            float values
        mean: float
            mean
        std: float
            standard deviation
        """
    return x*std + mean

def getDevice():
    print("CUDA available?",torch.cuda.is_available())
    assert(torch.cuda.is_available())
    device = torch.device("cuda")
    return device

def fastShuffle(series, num):
    length = series.shape[0]
    for i in range(num):
        series = series[np.random.permutation(length),:]
    return series
    