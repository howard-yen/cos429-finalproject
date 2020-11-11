import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as dset

random.seed(999)
torch.manual_seed(999)

# root directory for dataset
dataroot = "data/celeba"
# number of workers for dataloader
workers = 2
# batch size for training
batch_size = 128
# spatial size of training images
image_size = 64
# number of channels in training images
nc = 3
# size of generator input
nz = 100
# size of feature maps in generator
nfg = 64
# size of feature maps in discriminator
ndf = 64
# number of training epochs
num_epochs = 5
# learning rate
lr = 0.0002
# beta1 for Adam optimizer
beta1 = 0.5
# number of GPUs
ngpu = 0

dataset = dset.ImageFolder(root=dataroot, 
                           transform=transforms.Compse([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
