import torch
import torch.nn.functional as F

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

# define alphas 
def get_alphas(betas):
    return 1. - betas

def get_alphas_cumprod(alphas):
    return torch.cumprod(alphas, axis=0)

def get_alphas_cumprod_prev(alphas):
    return F.pad(get_alphas_cumprod(alphas)[:-1], (1, 0), value=1.0)

def get_sqrt_recip_alphas(alphas):
    return torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
def get_sqrt_alphas_cumprod(alphas):
    return torch.sqrt(get_alphas_cumprod(alphas))
def get_sqrt_one_minus_alphas_cumprod(alphas):
    return torch.sqrt(1. - get_alphas_cumprod(alphas))

# calculations for posterior q(x_{t-1} | x_t, x_0)
def get_posterior_variance(alphas, betas):
    return betas * (1. - get_alphas_cumprod_prev(alphas)) / (1. - get_alphas_cumprod(alphas))

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(x_start, t, betas, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    alphas = get_alphas(betas)
    sqrt_alphas_cumprod_t = extract(get_sqrt_alphas_cumprod(alphas), t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        get_sqrt_one_minus_alphas_cumprod(alphas), t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

if __name__ == "__main__":
    # Plot the cosine beta schedule and linear beta schedule
    import matplotlib.pyplot as plt
    timesteps = 100
    cosine_betas = cosine_beta_schedule(timesteps)
    linear_betas = linear_beta_schedule(timesteps)
    plt.plot(cosine_betas)
    plt.plot(linear_betas)
    plt.show()