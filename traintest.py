import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils

img_list = []
G_losses = []
D_losses = []

fixed_noise = torch.randn(64, 100, 1, 1, device="cpu") # 100=nz, device=device

def train(netG: nn.Module, netD: nn.Module, optG: torch.optim.Adam, optD: torch.optim.Adam, dataloader: DataLoader, criterion:nn.BCELoss, nz:int, epochs: int, device:str="cpu"):
    iters = 0
    for epoch in range(epochs):
        real_label = 1.
        fake_label = 0.

        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()

            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_cpu).view(-1)

            errD_real = criterion(output, label)
            errD_real.backward()

            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)

            output = netD(fake.detach()).view(-1)

            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_fake + errD_real

            optD.step()

            netG.zero_grad()
            label.fill_(real_label)

            output = netD(fake).view(-1)

            errG = criterion(output, label)
            errG.backward()

            D_G_z2 = output.mean().item()

            optG.step()

            if (iters % 500 == 0) or ((epoch == epochs - 1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
            iters += 1