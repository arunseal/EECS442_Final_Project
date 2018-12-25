import torch
from torch import nn
import numpy as np
import torch
from models.layers import Conv, Hourglass, Pool
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function, Variable
import imageio
import os
import sys
import cv2
import shutil
import imageio


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

    shutil.rmtree('../data/test/normal')
    os.makedirs('../data/test/normal')


    model =  torch.load('../models/hg_test')

    test = SurfaceNormalsDataset('../data/test', test = True)
    for idx,data in enumerate(test):
        image = data['image']
        pred = model(Variable(image.unsqueeze(0)))
        pred = pred.squeeze()
        
        pred = pred.data.numpy()
        pred = np.array(pred).astype(np.uint8)
        pred = pred.transpose(1,2,0)
        fname = str(idx)+'.png'

        imageio.imwrite('../data/test/normal/' + str(idx)+'.png', pred)



if __name__ == '__main__':
    main()