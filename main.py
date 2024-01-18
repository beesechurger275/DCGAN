import torch
from generator import Generator
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from config import nz, generator_load

device = "cuda" if torch.cuda.is_available() else "cpu"

generator = Generator().to(device)

generator.load_state_dict(torch.load(generator_load))

with torch.no_grad():
    img = generator(torch.randn(64, nz, 1, 1, device=device)).detach().cpu()

img = vutils.make_grid(img, padding=2, normalize=True)

plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(img, (1,2,0)))
plt.show()