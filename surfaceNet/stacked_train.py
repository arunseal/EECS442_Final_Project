from torch import nn
from torch.autograd import Function, Variable
import sys 
import time
import numpy as np
import cv2
import torch
from layers import Conv, Hourglass, Pool, MAELoss, StackedHourglass
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import imageio
import torch.optim as optim
import json
from utils import SurfaceNormalsDataset, get_validation_loss
from utils import timeStr, timeRemaining
import time
import argparse

parser = argparse.ArgumentParser(description='PyTorch based hourglass network training')
parser.add_argument('--gpu', type=int, default=1, metavar='G',
                    help='0 for CPU, 1 for GPU (default: 1)')
parser.add_argument('--model', type=str, default="", metavar='M',
                    help='Input model name.  If nothing given, creates hardcoded model')
parser.add_argument('--outputName', type=str, default="new_model", metavar='O',
                    help='Name which trained model will be saved as (default: new_model)')
parser.add_argument('--numEpochs', type=int, default=0, metavar='E',
                    help='Number of training epochs (default: 0')
parser.add_argument('--learn', type=float, default=1e-4, metavar='L',
                    help='Learning Rate (default: 0.0001')
parser.add_argument('--increase', type=int, default=32, metavar='I',
                    help='Hourglass layers increase parameter (default: 32)')
parser.add_argument('--batchSize', type=int, default=28, metavar='B',
                    help='Training batch size (default: 28)')
parser.add_argument('--numFilters', type=int, default=10, metavar='F',
                    help='Number of convolutional filters (default: 10)')
parser.add_argument('--numStacks', type=int, default=1, metavar='N',
                    help='Number of hourglasses to stack (default: 1)')
parser.add_argument('--hourglassSteps', type=int, default=4, metavar='S',
                    help='Number of times hourglass layer decreases in size (default: 4)')
parser.add_argument('--dataPath', type=str, default="../data", metavar='D',
                    help='Path to data folder (default: ../data')
args = parser.parse_args()


def main():

    train = SurfaceNormalsDataset(args.dataPath + '/actual_training', test = False, normalize=True)
    trainloader = torch.utils.data.DataLoader(train, batch_size=args.batchSize, shuffle=True)

    valid = SurfaceNormalsDataset(args.dataPath + '/validation', test = False, normalize=True)

    if args.model == "":
        model = StackedHourglass(args.hourglassSteps, args.numFilters, args.increase, args.numStacks)
    else:
        model = torch.load("../models/" + args.model)
    
    if args.gpu:
        torch.cuda.set_device(0)
        model = model.cuda()


    # criterion = torch.nn.MSELoss(size_average=False)
    criterion = MAELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn)
    
    start = time.time()
    for epoch in range(args.numEpochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs = data['image']
            labels = data['normal']

            # wrap them in Variable
            if args.gpu:
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()     
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss_list = []
            for j in range(model.numStacks):
                loss_list.append(criterion(outputs[j], labels))
            loss = torch.cat(loss_list).sum()/model.numStacks
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            denom = (10 * 3 * 128 * 128 * args.batchSize)
            denom = 10
            if i % 10 == 9:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.8f' %
                      (epoch + 1, i + 1, running_loss / denom))
                running_loss = 0.0
                print("Time taken: " + timeStr(time.time() - start))
                print("Time remaining: " + timeRemaining(start, time.time(), epoch, args.numEpochs, i, len(trainloader)))
                torch.save(model, '../models/' + args.outputName)

                valid_loss = get_validation_loss(model, valid, criterion, args)
                print('validation loss:', valid_loss.data[0], '\n')
                sys.stdout.flush()

        torch.save(model, '../models/' + args.outputName)


    print('Finished Training')










if __name__ == '__main__':
    main()
