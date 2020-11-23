import random

import numpy as np
import torch
import torch.nn as nn

random.seed(999)
torch.manual_seed(999)

#device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
device = "cpu"
# TODO: make sure to .to(device) the class later, and also set up gpu

# height and width of input image
img_size = 32
# number of channels
nc0 = 1
nc1 = 4
nc2 = 8

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.conv1 = nn.Conv2d(nc0, nc1, 3, padding=1)
        self.conv2 = nn.Conv2d(nc1, nc2, 3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(nc1, nc0, 3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(nc2, nc1, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(nc1)
        self.batchnorm2 = nn.BatchNorm2d(nc2)
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax()

    def forward(self, input):
        self.conv1(input)
        self.batchnorm1()
        self.relu()
        self.conv2()
        self.pool()
        self.batchnorm1()
        self.relu()
        self.pool()
        self.unpool()
        self.deconv2()
        self.batchnorm2()
        self.relu()
        self.deconv1()
        self.batchnorm1()
        self.relu()
        self.softmax()
