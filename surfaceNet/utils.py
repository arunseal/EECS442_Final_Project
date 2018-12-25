from torch import nn
from torch.autograd import Function, Variable
import sys 
import time
import numpy as np
import cv2
import torch
from layers import Conv, Hourglass, Pool
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import imageio
import torch.optim as optim
import json

def timeStr(t):
    units = " seconds"
    if(t > 60):
        t /= 60
        units = " minutes"
    if(t > 60):
        t /= 60
        units = " hours"
    return "{0:.2f}".format(t) + units

def timeRemaining(start, current, epoch, numEpochs, batch, numBatches):
    t = current - start
    batchProgress = batch/numBatches
    totalProgress = (epoch+batchProgress)/numEpochs
    totalTime = t/totalProgress
    return timeStr(totalTime - t) 

def to_tensor(image):
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image)


class SurfaceNormalsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, test = False, transform=None, normalize = False):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.test = test
        self.normalize = normalize
        fnames = os.listdir(os.path.join(self.root_dir,'color'))
        fnames = sorted(fnames, key = lambda x: int(x.strip('.png')))
        self.fnames = fnames

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir,'color')))

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        image = imageio.imread(os.path.join(self.root_dir, 'color', fname))
        image = np.array(image).astype(np.float32)

        if self.normalize:
            image = ((image / 255.0) - 0.5) * 2
        
        sample = {'image': to_tensor(image)}

        if not self.test:
            normal = imageio.imread(os.path.join(self.root_dir, 'normal', fname))
            normal = np.array(normal).astype(np.float32)
            mask = imageio.imread(os.path.join(self.root_dir, 'mask', fname))
            normal = cv2.bitwise_and(normal, normal, mask = mask)
            if self.normalize:
                pass
                # normal = ((normal / 255.0) -0.5) * 2

            sample['normal'] = to_tensor(normal)

            


        if self.transform:
            sample = self.transform(sample)

        return sample



def get_validation_loss(model, dataset, criterion, args):
    losses = []
    divisor = (3 * 128 * 128)
    divisor = (1)

    nums = np.arange(2000)
    np.random.shuffle(nums)
    nums = nums[:25]


    for num in nums:
        data = dataset[num]

        # get the inputs
        inputs = data['image']
        labels = data['normal']

        inputs = inputs.unsqueeze(0)
        labels = labels.unsqueeze(0)

        # wrap them in Variable
        if args.gpu:
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()     
        else:
            inputs, labels = Variable(inputs).cpu(), Variable(labels).cpu()
            model = model.cpu()
       
        outputs = model(inputs)
        if(len(outputs.data.shape) == 5):
            outputs = outputs[-1,:,:,:,:]

        loss = criterion(outputs, labels)
        losses.append(loss)


    return np.mean(losses)/divisor