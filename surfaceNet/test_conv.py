from torch import nn
from torch.autograd import Function, Variable
import sys 
import time
import numpy as np
import cv2
import torch
from layers import Conv, Hourglass, Pool, MAELoss, cosDist
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


def main():

    batchSize = 25
    gpu = True

    train = SurfaceNormalsDataset('../data/train', test = False, normalize=True)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batchSize, shuffle=True)

    #valid = SurfaceNormalsDataset('../data/validation', test = False, normalize=True)



    model = torch.load('../models/model')
    # model = torch.nn.Sequential(
    #     Conv(3,512),
    #     Conv(512,512),
    #     Conv(512,3),
    #     )

    if gpu:
        model = model.cuda()



    # criterion = torch.nn.MSELoss(size_average=False)
    criterion = MAELoss()
    # criterion = cosDist()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    start = time.time()
    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs = data['image']
            labels = data['normal']

            # wrap them in Variable
            if gpu:
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()     
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            #print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            # print statistics
            running_loss += loss.data[0]
            if i % 10 == 9:    # print every 10 mini-batches
                divisor = (10 * 3 * 128 * 128 * batchSize)
                divisor = 10
                print('[%d, %5d] loss: %.8f' %
                      (epoch + 1, i + 1, running_loss / (divisor)))
                running_loss = 0.0
                print("Time taken: " + timeStr(time.time() - start))
                print("Time remaining: " + timeRemaining(start, time.time(), epoch, 10, i, len(trainloader)))
                torch.save(model, '../models/model')

                #valid_loss = get_validation_loss(model, valid, criterion, args)
                #print('validation loss:', valid_loss.data[0], '\n')
                sys.stdout.flush()

        torch.save(model, '../models/model')


    print('Finished Training')






if __name__ == '__main__':
    main()
