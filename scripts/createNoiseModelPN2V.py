import warnings
warnings.filterwarnings('ignore')
import torch
dtype = torch.float
device = torch.device("cuda:0") 
from torch.distributions import normal
import matplotlib.pyplot as plt, numpy as np, pickle
from scipy.stats import norm
from tifffile import imread
import sys
sys.path.append('../../../')
from pn2v import *
import pn2v.gaussianMixtureNoiseModel
import pn2v.histNoiseModel
from pn2v.utils import *
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--outPath", help="The path to your training data")
parser.add_argument("--name", help="The name of ypu noise model")
parser.add_argument("--signal", help="The path to your clean signal")
parser.add_argument("--observation", help="The path to your noisy observation")
parser.add_argument("--n_gaussian", help="Number of gaussians to use for Gaussian Mixture Model", default=3, type=int)
parser.add_argument("--n_coeff", help="No. of polynomial coefficients for parameterizing the mean, standard deviation and weight of Gaussian components", default=3, type=int)

parser.add_argument("--minSigma", help="The minimum allowed standatd deviation a gaussian component can have", default=2, type=float)
parser.add_argument("--minSignalPrec", help="Signals below this percentile will be disregarded", default=0.5, type=float)
parser.add_argument("--maxSignalPrec", help="Signals above this percentile will be disregarded", default=99.5, type=float)

parser.add_argument("--learningRate", help="The learning rate", default=0.05, type=float)
parser.add_argument("--batchSize", help="Training batch size in pixels", default=25000, type=int)

parser.add_argument("--n_epochs", help="Number of training epochs", default=2000, type=int)

if len(sys.argv)==1:
    print("exit")
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()
print(args)


n_gaussian = args.n_gaussian # Number of gaussians to use for Gaussian Mixture Model
n_coeff = args.n_coeff # No. of polynomial coefficients for parameterizing the mean, standard deviation and weight of Gaussian components.

observation= imread(str(args.observation)) # Load the appropriate observation data
signal= imread(str(args.signal)) # Load the appropriate signal data

nameGMMNoiseModel = 'GMMNoiseModel_'+str(args.name)+'_'+str(n_gaussian)+'_'+str(n_coeff)

minVal, maxVal = np.min(observation), np.max(observation)
min_signal=np.percentile(signal, args.minSignalPrec)
max_signal=np.percentile(signal, args.maxSignalPrec)
print("Minimum Signal Intensity is", min_signal)
print("Maximum Signal Intensity is", max_signal)

gaussianMixtureNoiseModel = pn2v.gaussianMixtureNoiseModel.GaussianMixtureNoiseModel(min_signal = min_signal,
                                                                                     max_signal =max_signal,
                                                                                     path=str(args.outPath), 
                                                                                     weight = None,
                                                                                     n_gaussian = n_gaussian, 
                                                                                     n_coeff = n_coeff,
                                                                                     min_sigma = args.minSigma,
                                                                                     device = device)

gaussianMixtureNoiseModel.train(signal, 
                                observation,
                                batchSize = args.batchSize, 
                                n_epochs = args.n_epochs, 
                                learning_rate=args.learningRate,
                                name = nameGMMNoiseModel)
