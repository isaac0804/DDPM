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

if __name__ == "__main__":
    pe = SinusoidalPositionEmbeddings(100)
    # Plot the positional encoding
    import matplotlib.pyplot as plt
    time = torch.arange(0, 200, dtype=torch.float32)
    embeddings = pe(time)
    plt.imshow(embeddings)
    plt.show()