import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

random.seed(999)
torch.manual_seed(999)

#device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
device = "cpu"
# TODO: make sure to .to(device) the class later, and also set up gpu

# path to font list
fonts_csv = "fonts.csv"
# root directory for dataset
dataroot = "images"
# number of workers for dataloader
workers = 0
# number of epochs
num_epochs = 5
# batch size for training
batch_size = 4
# height and width of input image
img_size = 48
# number of channels
nc0 = 1
nc1 = 4
nc2 = 8
# learning rate
lr = 0.0002
# beta1 for Adam
beta1 = 0.5

class FontDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.fontlist = pd.read_csv(csv_file, sep=' ')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.fontlist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path1 = os.path.join(self.root_dir, "a/", f"{idx}.npy")
        img_path2 = os.path.join(self.root_dir, "b/", f"{idx}.npy")

        img1 = np.load(img_path1)
        img2 = np.load(img_path2)

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        sample = {'c1': img1, 'c2': img2}

        return sample

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x, idx1 = self.pool(x)

        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x, idx2 = self.pool(x)

        x = self.unpool(x, idx2)
        x = self.deconv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)

        x = self.unpool(x, idx1)
        x = self.deconv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.softmax(x)
        return x

dataset = FontDataset(csv_file=fonts_csv, 
                      root_dir=dataroot, 
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(0.5, 0.5),
                      ]))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

encdec = EncoderDecoder()
criterion = nn.L1Loss()
optimizer = optim.Adam(encdec.parameters(), lr=lr, betas=(beta1, 0.999))

# training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        # zero out gradients
        encdec.zero_grad()
        output = encdec(data['c1'])
        loss = criterion(output, data['c2'])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f"Epoch {epoch+1}, Iteration {i+1}, Loss {running_loss}")
            running_loss = 0.0

print("Done")
