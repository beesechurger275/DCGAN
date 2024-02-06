import torch
from generator import Generator
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import math
from config import device, nz, generator_load, num_generated_images

generator = Generator().to(device)

generator.load_state_dict(torch.load(generator_load))

with torch.no_grad():
    img: torch.Tensor = generator(torch.randn(num_generated_images, nz, 1, 1, device=device)).detach().cpu()

rows = int(math.sqrt(num_generated_images))

img = vutils.make_grid(img, padding=2, normalize=True, nrow=rows)

plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(img, (1,2,0)))
plt.show()