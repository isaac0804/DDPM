import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from composer.models import UNet
from tqdm import tqdm


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

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        get_sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(get_sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(get_posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
    

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