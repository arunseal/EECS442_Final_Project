from torch import nn
from torch.autograd import Function, Variable
import sys 
import time
import numpy as np
import cv2
import torch

Pool = nn.MaxPool2d

def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, (1./n)**0.5) 

class Full(nn.Module):
    def __init__(self, inp_dim, out_dim, bn = False, relu = False):
        super(Full, self).__init__()
        self.fc = nn.Linear(inp_dim, out_dim, bias = True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.fc(x.view(x.size()[0], -1))
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x




class Conv_with_mask(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x







class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=128):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Conv(f, f, 3, bn=bn)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Conv(f, nf, 3, bn=bn)
        # Recursive hourglass
        if n > 1:
            self.low2 = Hourglass(n-1, nf, bn=bn)
        else:
            self.low2 = Conv(nf, nf, 3, bn=bn)
        self.low3 = Conv(nf, f, 3)
        #self.up2  = nn.UpsamplingNearest2d(scale_factor=2)
        self.up2  = nn.Upsample(scale_factor=2)

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2


class StackedHourglass(nn.Module):
    def __init__(self, hourglassSteps, numFilters, increase, numStacks, bn=None):
        super(StackedHourglass, self).__init__()
        self.numStacks = numStacks
        self.conv1 = nn.ModuleList([Conv(3, numFilters) for _ in range(numStacks)])
        self.hourglass = nn.ModuleList([Hourglass(hourglassSteps, numFilters, bn, increase=increase) for _ in range(numStacks)])
        self.conv2 = nn.ModuleList([Conv(numFilters, numFilters) for _ in range(numStacks)])
        self.conv3 = nn.ModuleList([Conv(numFilters, 3) for _ in range(numStacks)])

    def forward(self, x):
        intermediate = []
        for i in range(self.numStacks):
            x = self.conv1[i](x)
            x = self.hourglass[i](x)
            x = self.conv2[i](x)
            x = self.conv3[i](x)
            intermediate.append(x)
        return torch.stack(intermediate)


class cosDist(nn.Module):
    def __init__(self):
        super(cosDist, self).__init__()

    def forward(self, preds, normals):
        batch_size, _, length, width = preds.data.shape
        
        loss = 0
        total_pixels = 0
        for idx, (pred,actual) in enumerate(zip(preds,normals)):


            mask = (torch.sum(actual, dim =0) != 0)
            inv_mask = mask.data^1
            tmp_pixels = torch.sum(mask.double())
            total_pixels += tmp_pixels.data[0]

            pred = ((pred / 255.0) - 0.5) * 2
            actual = ((actual / 255.0) - 0.5) * 2

            a11 = torch.sum(pred * pred, dim=0)
            a22 = torch.sum(actual * actual, dim=0)
            a12 = torch.sum(pred * actual, dim=0)


            cos_sim = a12 / torch.sqrt(a11 * a22)
            # handles NaNs
            # commented out b/c not differentiable
            # cos_dist[cos_dist != cos_dist] = -1

            cos_dist = 1.0 - cos_sim

            cos_dist[inv_mask] = 0.0
            loss += torch.sum(cos_dist)

        #print('total_pixels:', total_pixels)
        loss = loss / float(total_pixels)   # normalize by batch size
        return loss

def where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, preds, normals):
        batch_size, _, length, width = preds.data.shape
        
        loss = 0
        total_pixels = 0
        #dim = 2
        dim = 0
        for idx, (pred,actual) in enumerate(zip(preds,normals)):

            # print(torch.min(pred))
            # print(actual.data.shape)

            mask = (torch.sum(actual, dim = dim) != 0)
            tmp_pixels = torch.sum(mask.double())
            total_pixels += tmp_pixels.data[0]

            pred = ((pred / 255.0) - 0.5) * 2
            actual = ((actual / 255.0) - 0.5) * 2

            a11 = torch.sum(pred * pred, dim=dim)[mask]
            a22 = torch.sum(actual * actual, dim=dim)[mask]
            a12 = torch.sum(pred * actual, dim=dim)[mask]


            cos_dist = a12 / torch.sqrt(a11 * a22)
            # handles NaNs
            # commented out b/c not differentiable
            # cos_dist[cos_dist != cos_dist] = -1
            numnan = torch.sum(cos_dist.data[cos_dist.data != cos_dist.data])
            if numnan > 0:
                print('you got a nan')
                exit(1)
            # cos_dist = where((cos_dist!=cos_dist).float(), -1.0, cos_dist)

            cos_dist = torch.clamp(cos_dist, -0.9999, 0.9999)
            angle_error = torch.acos(cos_dist)
            loss += torch.sum(angle_error)

        loss = loss / float(total_pixels)   # normalize by pixels in mask
        return loss

