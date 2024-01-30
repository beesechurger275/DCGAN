from config import *
from generator import Generator
from discriminator import Discriminator
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transform
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

dataset = datasets.ImageFolder(dataroot, 
transform=transform.Compose([
    transform.Resize(image_size),
    transform.CenterCrop(image_size),
    transform.ToTensor(),
    transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

if input("Load model? (y/n)").lower() == "y":
    generator.load_state_dict(torch.load(generator_load))
    discriminator.load_state_dict(torch.load(discriminator_load))

generator.apply(weights_init)
discriminator.apply(weights_init)

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

loss_fn = nn.BCELoss()
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1,0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1,0.999))

img_list = []
G_loss = []
D_loss = []

real_label = 1.
fake_label = 0.

try:
    print("begin train loop...")
    for epoch in range(num_epochs):
        iters = 0
        for i, data in enumerate(dataloader, 0):
            discriminator.zero_grad()

            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            out = discriminator(real_cpu).view(-1)
            err_d_real = loss_fn(out, label)
            err_d_real.backward()
            d_x = out.mean().item()

            fake = generator(torch.randn(b_size, nz, 1, 1, device=device))
            label.fill_(fake_label)

            out = discriminator(fake.detach()).view(-1)
            err_d_fake = loss_fn(out, label)
            err_d_fake.backward()
            d_g_z1 = out.mean().item()
            err_d = err_d_real + err_d_fake
            optimizerD.step()

            generator.zero_grad()
            label.fill_(real_label)
            out = discriminator(fake).view(-1)
            err_g = loss_fn(out, label)
            err_g.backward()
            d_g_z2 = out.mean().item()
            optimizerG.step()
            iters += 1

            if iters % 500 == 0 or iters == len(dataloader):
                print(f"[{epoch + 1}/{num_epochs}], [{iters}/{len(dataloader)}], err_g: {err_g:2f}, err_d: {err_d:2f} err_d_real: {err_d_real:2f}, err_d_fake: {err_d_fake:2f}")
                
                """ # Uncomment to view model progress
                with torch.no_grad():
                    prev = generator(fixed_noise).detach().cpu()
                    img_list.append(prev)
                
                plt.figure(figsize=(8,8))
                plt.axis("off")
                plt.title("Generated Images")
                plt.imshow(np.transpose(vutils.make_grid(prev, padding=2, normalize=True), (1,2,0)))
                plt.show()
                """
                

except KeyboardInterrupt:
    pass

except Exception as e:
    print("exception!")
    print(e)

input_ = input("save model? (y/n) ").lower()
if input_ != "n" and input_ != "no":
    torch.save(generator.state_dict(), generator_save)
    torch.save(discriminator.state_dict(), discriminator_save)