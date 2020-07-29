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
import matplotlib.pyplot as plt


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


def plotProbabilityDistribution(signalBinIndex, histogram, gaussianMixtureNoiseModel, min_signal, max_signal, n_bin, device):
    """Plots probability distribution P(x|s) for a certain ground truth signal.
       Predictions from both Histogram and GMM-based Noise models are displayed for comparison.
        Parameters
        ----------
        signalBinIndex: int
            index of signal bin. Values go from 0 to number of bins (`n_bin`).
        histogram: numpy array
            A square numpy array of size `nbin` times `n_bin`.
        gaussianMixtureNoiseModel: GaussianMixtureNoiseModel
            Object containing trained parameters.
        min_signal: float
            Lowest pixel intensity present in the actual sample which needs to be denoised.
        max_signal: float
            Highest pixel intensity present in the actual sample which needs to be denoised.
        n_bin: int
            Number of Bins.
        device: GPU device
        """
    histBinSize=(max_signal-min_signal)/n_bin
    querySignal_numpy= (signalBinIndex/float(n_bin)*(max_signal-min_signal)+min_signal)
    querySignal_numpy +=histBinSize/2
    querySignal_torch = torch.from_numpy(np.array(querySignal_numpy)).float().to(device)

    queryObservations_numpy=np.arange(min_signal, max_signal, histBinSize)
    queryObservations_numpy+=histBinSize/2
    queryObservations = torch.from_numpy(queryObservations_numpy).float().to(device)
    pTorch=gaussianMixtureNoiseModel.likelihood(queryObservations, querySignal_torch)
    pNumpy=pTorch.cpu().detach().numpy()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.xlabel('Observation Bin')
    plt.ylabel('Signal Bin')
    plt.imshow(histogram**0.25, cmap='gray')
    plt.axhline(y=signalBinIndex+0.5, linewidth=5, color='blue', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(queryObservations_numpy, histogram[signalBinIndex, :]/histBinSize, label='GT Hist: bin ='+str(signalBinIndex), color='blue', linewidth=2)
    plt.plot(queryObservations_numpy, pNumpy, label='GMM : '+' signal = '+str(np.round(querySignal_numpy,2)), color='red',linewidth=2)
    plt.xlabel('Observations (x) for signal s = ' + str(querySignal_numpy))
    plt.ylabel('Probability Density')
    plt.title("Probability Distribution P(x|s) at signal =" + str(querySignal_numpy))
    plt.legend()
