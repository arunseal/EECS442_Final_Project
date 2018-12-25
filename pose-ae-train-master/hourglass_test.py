from torch import nn
from torch.autograd import Function, Variable
import sys 
import time
import numpy as np
import cv2
import torch
from models.layers import Conv, Hourglass, Pool
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import imageio
import torch.optim as optim
import json


def to_tensor(image):
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image)


class SurfaceNormalsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, test = False, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir,'color')))

    def __getitem__(self, idx):
        image = imageio.imread(os.path.join(self.root_dir, 'color', str(idx)+'.png'))
        image = np.array(image).astype(np.float32)
        
        sample = {'image': to_tensor(image)}

        if not self.test:
            normal = imageio.imread(os.path.join(self.root_dir, 'normal', str(idx)+'.png'))
            normal = np.array(normal).astype(np.float32)
            mask = imageio.imread(os.path.join(self.root_dir, 'mask', str(idx)+'.png'))
            normal = cv2.bitwise_and(normal, normal, mask = mask)
            sample['normal'] = to_tensor(normal)

        if self.transform:
            sample = self.transform(sample)

        return sample





def main():

    print('test')

    train = SurfaceNormalsDataset('../data/train', test = False)
    trainloader = torch.utils.data.DataLoader(train, batch_size=25, shuffle=True)

    model = torch.nn.Sequential(
        Conv(3,10),
        Hourglass(4,10,bn=None,increase=20),
        Conv(10,10),
        Conv(10,3),
        )

    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(3):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs = data['image']
            labels = data['normal']

            # wrap them in Variable
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
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / (10*25*128*128*3)))
                running_loss = 0.0

            if i == 100:
                pass
                #break

    torch.save(model, '../models/hg_test')


    print('Finished Training')










if __name__ == '__main__':
    main()
