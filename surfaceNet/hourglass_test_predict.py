import torch
from torch import nn
import numpy as np
import torch
from layers import Conv, Hourglass, Pool
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function, Variable
import imageio
import os
import sys
import shutil
import imageio
from utils import SurfaceNormalsDataset
import argparse

parser = argparse.ArgumentParser(description='PyTorch based hourglass network inference')
parser.add_argument('--gpu', type=int, default=0, metavar='G',
                    help='0 for CPU, 1 for GPU (default: 0)')
parser.add_argument('--model', type=str, default="", metavar='M',
                    help='Model Name')
parser.add_argument('--average', type=int, default=0, metavar='V',
                    help='If 1, averages output with output of reversed image for inference (default: 0)')
parser.add_argument('--axis', type=int, default=2, metavar='A',
                    help='Axis to flip over if averaging is True. 1: Up/Down, 2: Left/Right (default: 2)')
args = parser.parse_args()

def main():

    in_dir = '../data/test/'
    #in_dir = '../data/train/'

    shutil.rmtree(in_dir + 'pred')
    os.makedirs(in_dir + 'pred')

    modelPath = "../models/" + args.model
    model =  torch.load(modelPath, map_location='cpu')
    if args.gpu:
        torch.cuda.set_device(0)
        model = model.cuda()


    test = SurfaceNormalsDataset(in_dir, test = True, normalize = True)
    for idx in range(len(test)):
        data = test[idx]
        

        image = data['image']
        # x = image
        image = Variable(image.unsqueeze(0))

        if args.gpu:
            image = image.cuda()

        pred = model(image)
        pred = pred.data

        if args.average:
            imageR = data['image']
            imageR = np.flip(imageR, axis=args.axis).copy()
            # imageio.imwrite(in_dir + 'pred/img' + str(idx)+'.png', x.numpy().transpose(1,2,0))
            # imageio.imwrite(in_dir + 'pred/imgR' + str(idx)+'.png', imageR.transpose(1,2,0))
            imageR = Variable(torch.from_numpy(imageR))
            imageR = imageR.unsqueeze(0)

            if args.gpu:
                imageR = imageR.cuda()

            predR = model(imageR)
            predR = predR.data

            # pred = pred.squeeze().numpy().transpose(1,2,0)
            # predR = predR.squeeze().numpy().transpose(1,2,0)
            # print(pred.shape)
            # print(predR.shape)
            # imageio.imwrite(in_dir + 'pred/pred' + str(idx)+'.png', pred)
            # imageio.imwrite(in_dir + 'pred/predR' + str(idx)+'.png', predR)
        
        if(len(pred.shape) == 5):
            pred = pred[-1,:,:,:,:]
            if args.average:
                predR = predR[-1,:,:,:,:]
        
        pred = pred.squeeze()
        if args.gpu:
            pred = pred.cpu()
            if args.average:
                predR = predR.cpu()
        
        pred = pred.numpy()

        # unnormalize
        # pred = ((pred/2) + 0.5)*255.0
        pred = np.clip(pred, 0, 255)

        pred = np.array(pred).astype(np.uint8)
        pred = pred.transpose(1,2,0)

        if args.average:
            predR = predR.squeeze()
            predR = predR.numpy()

            # unnormalize
            # pred = ((pred/2) + 0.5)*255.0
            predR = np.clip(predR, 0, 255)

            predR = np.array(predR).astype(np.uint8)
            predR = np.flip(predR, axis=args.axis).copy()
            predR = predR.transpose(1,2,0)
            
            pred = ((pred.astype(np.float) + predR.astype(np.float))/2).astype(np.uint8)

        fname = str(idx)+'.png'

        imageio.imwrite(in_dir + 'pred/' + str(idx)+'.png', pred)




if __name__ == '__main__':
    main()
