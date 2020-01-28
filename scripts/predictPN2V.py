import os
import sys
import argparse
import glob
sys.path.append('../')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--netPath", help="Directory in which we find you network. If None, we use the dataPath ", default=None)
parser.add_argument("--name", help="name of your network", default='N2V')
parser.add_argument("--dataPath", help="The path to your data")
parser.add_argument("--fileName", help="name of your data file", default="*.tif")
parser.add_argument("--output", help="The path to which your data is to be saved", default='.')
parser.add_argument("--tileSize", help="width/height of tiles used to make it fit GPU memory", default=256, type=int)
parser.add_argument("--tileOvp", help="overlap of tiles used to make it fit GPU memory", default=48, type=int)
parser.add_argument("--noiseModel", help="path to the .npz file containing the parametric noise model", default='noiseModel.npz')

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()

print(args.output)

# We import all our dependencies.
import numpy as np
from matplotlib import pyplot as plt
from tifffile import imread
from tifffile import imwrite

import torch
from unet.model import UNet
from pn2v.utils import denormalize
from pn2v.utils import normalize
from pn2v import utils
from pn2v import prediction
import pn2v.training
from pn2v import gaussianMixtureNoiseModel


device=utils.getDevice()
path=args.dataPath


####################################################
#           PREPARE Noise Model
####################################################

params= np.load(args.noiseModel)
noiseModel = gaussianMixtureNoiseModel.GaussianMixtureNoiseModel(params = params, device = device)


####################################################
#           LOAD NETWORK
####################################################

netPath=path
if args.netPath is not None:
    netPath= args.netPath
net=torch.load(netPath+"/last_"+args.name+".net")


####################################################
#           PROCESS DATA
####################################################


files=glob.glob(path+args.fileName)

for f in files:
    print('loading',f)
    image= (imread(f).astype(np.float32))
    if len(image.shape)<3:
        image=image[np.newaxis,...]


    means=np.zeros(image.shape)
    mseEst=np.zeros(image.shape)
    for i in range (image.shape[0]):
        im=image[i,...]
        # processing image
        means[i,...], mseEst[i,...] = prediction.tiledPredict(im, net ,ps=args.tileSize, overlap=args.tileOvp,
                                            device=device,
                                            noiseModel=noiseModel)

    if im.shape[0]==1:
        means=means[0]
        mseEst=mseEst[0]

    outpath=args.output
    filename=os.path.basename(f).replace('.tif','_MMSE-PN2V.tif')
    outpath=os.path.join(outpath,filename)
    print('writing',outpath, mseEst.shape)
    imwrite(outpath, mseEst.astype(np.float32), imagej=True)

    outpath=args.output
    filename=os.path.basename(f).replace('.tif','_Prior-PN2V.tif')
    outpath=os.path.join(outpath,filename)
    print('writing',outpath, means.shape)
    imwrite(outpath, means.astype(np.float32), imagej=True)
