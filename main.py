import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as Dataset
import torchvision.transforms.v2 as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from generator import Generator
from discriminator import Discriminator
from traintest import train

seed = 9999

random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

dataroot = "data/img_align_celeba/"
workers = 0
batch_size = 128
image_size=64
nc = 3
nz = 100 # Size of z latent vector (i.e. size of generator input)
ngf = 64 # Size of feature maps in generator
ndf = 64 # Size of feature maps in discriminator
epochs = 5
lr = 1e-4
beta1 = 0.5 # hyperperam for Adam optimizer
ngpu = 0

dataset = Dataset.ImageFolder(root=dataroot, 
    transform=transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ]))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

device = "cpu" if not torch.cuda.is_available() else "cuda"

real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Sample training images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
#plt.show()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0, 0.2) # DCGAN init
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator = Generator(nz, ngf, nc).to(device)
generator.apply(weights_init)

discriminator = Discriminator(nc, ndf).to(device)
discriminator.apply(weights_init)

criterion = nn.BCELoss()
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_label = 1.
fake_label = 0.

optG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

train(generator, discriminator, optG, optD, dataloader, criterion, nz, 3, "cpu")