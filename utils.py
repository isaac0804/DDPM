import torch.nn as nn

def exists(x):
    return x is not None

def default(val, d):
    return val if exists(val) else d

class Residual(nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn
    
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    
def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)