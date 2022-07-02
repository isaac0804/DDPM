import torch
import torch.nn as nn
import math
from composer.models import UNet


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Position Embedding in Temporal Dimension during diffusion process
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings =  time[:, None] * embeddings[None, :]
        embeddings =  torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings

def cosine_beta_schedule(timesteps, s=0.008):
    """
    B_0 < B_1 < ... < B_T
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, s=0.008):
    """
    B_T - B_{T-1} = ... = B_1 - B_0
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

if __name__ == "__main__":
    """
    pe = SinusoidalPositionEmbeddings(100)
    # Plot the positional encoding
    import matplotlib.pyplot as plt
    time = torch.arange(0, 200, dtype=torch.float32)
    embeddings = pe(time)
    plt.imshow(embeddings)
    plt.show()
    """
    # Plot the cosine beta schedule and linear beta schedule
    import matplotlib.pyplot as plt
    timesteps = 100
    cosine_betas = cosine_beta_schedule(timesteps)
    linear_betas = linear_beta_schedule(timesteps)
    plt.plot(cosine_betas)
    plt.plot(linear_betas)
    plt.show()