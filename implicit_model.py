import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter

from composer import Trainer
from composer import ComposerModel
from composer.callbacks import LRMonitor
from composer.loggers import FileLogger, LogLevel, WandBLogger
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler

from tqdm import tqdm
from matplotlib import animation
from matplotlib import pyplot as plt

from data import get_cifar10
from callbacks import ImplicitSamplingCallback

# Implicit Layer Implementation

class ResnetImplicitLayer(nn.Module):
    def __init__(self, latent_dim, inner_dim, kernel_size=3, num_groups=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.inner_channels = inner_dim

        self.conv1 = nn.Conv2d(latent_dim, inner_dim, kernel_size, padding=kernel_size//2, groups=num_groups, bias=False)
        self.conv2 = nn.Conv2d(inner_dim, latent_dim, kernel_size, padding=kernel_size//2, groups=num_groups, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, inner_dim)
        self.norm2 = nn.GroupNorm(num_groups, latent_dim)
        self.norm3 = nn.GroupNorm(num_groups, latent_dim)

    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(self.conv2(y) + x)))

def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0, sampling=False):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    if sampling:
        images = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))

        if sampling:
            images.append(X[:,k%m].view_as(x0))

        if (res[-1] < tol):
            break
    return (X[:,k%m].view_as(x0), res) if not sampling else (X[:,k%m].view_as(x0), images)

class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self, x, sampling=False):
        with torch.no_grad():
            z = self.solver(lambda z: self.f(z, x), torch.zeros_like(x), sampling=sampling, **self.kwargs)
            if not sampling:
                z, self.foward_res = z
            else:
                z, images = z
                z = self.f(z, x)
                images.append(z)
                return images
        z = self.f(z, x)

        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)
        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y: autograd.grad(f0, z0, grad_outputs=y, retain_graph=True)[0] + grad, grad, **self.kwargs)
            return g
        z.register_hook(backward_hook)
        return z

class ResnetImplicitModel(nn.Module):
    def __init__(self, channels, latent_dim, inner_dim, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, latent_dim, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(latent_dim)

        f = ResnetImplicitLayer(latent_dim, inner_dim, kernel_size=3, num_groups=num_groups)
        self.deq = DEQFixedPoint(f, anderson, max_iter=50, tol=1e-2)

        self.norm2 = nn.BatchNorm2d(latent_dim)
        self.conv2 = nn.Conv2d(latent_dim, channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        x = self.norm1(self.conv1(x)).contiguous()
        x = self.norm2(self.deq(x))
        x = self.conv2(x)
        return x

class DDIM(ComposerModel):
    def __init__(self, model, batch_size, image_size, channels, loss_type="huber"):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = channels
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss()
        elif loss_type == "huber":
            self.criterion = nn.SmoothL1Loss()
        else:
            raise NotImplementedError()

    def forward(self, batch):
        images, _ = batch # Excluding labels
        noise = torch.randn_like(images)
        return self.model(noise)

    def loss(self, generated_images, batch):
        target, _ = batch
        loss = self.criterion(generated_images, target)
        return loss

    @torch.no_grad()
    def sample(self, num_samples=4):
        noise = torch.randn(num_samples, self.channels,
                            self.image_size, self.image_size, device=self.device)
        noise = self.model.norm1(self.model.conv1(noise)).contiguous()
        noise = self.model.deq(noise, sampling=True)
        noise = torch.cat(noise, dim=0)
        noise = self.model.conv2(self.model.norm2(noise))
        # convert to [num_samples, batch_size, image_size, image_size, channels]
        noise = noise.view(-1, num_samples, self.channels, self.image_size, self.image_size).permute(1, 0, 3, 4, 2) 
        print(noise.shape)
        return noise

if __name__ == "__main__":

    batch_size = 64
    image_size = 32
    channels = 1
    latent_dim = 16
    inner_dim = 64
    num_groups = 8
    initial_lr = 1e-3
    final_lr = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, _ = get_cifar10(gray_scale=(channels==1))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    model = ResnetImplicitModel(channels=channels, latent_dim=latent_dim, inner_dim=inner_dim, num_groups=num_groups).to(device)
    model = DDIM(model, batch_size, image_size, channels, loss_type="huber")
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    run_name = "ddim-gray"

    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        max_duration="100ep",

        train_dataloader=train_loader,
        device="gpu" if torch.cuda.is_available() else "cpu",

        run_name=run_name,
        save_folder=f"runs/{run_name}/checkpoints",
        save_interval="5ep",

        # Scheduler
        schedulers=[
            CosineAnnealingWithWarmupScheduler(
                t_warmup="5ep", alpha_f=final_lr)
        ],

        # Callbacks
        callbacks=[
            ImplicitSamplingCallback(),
            LRMonitor()
        ],

        # Loggers
        loggers=[
            WandBLogger(),
            FileLogger(
                filename="runs/{run_name}/logs-rank{rank}.txt",
                buffer_size=1,
                capture_stderr=False,
                capture_stdout=False,
                log_level=LogLevel.EPOCH,
                log_interval=1,
                flush_interval=2
            )
            # TensorboardLogger(log_dir="runs", flush_interval=1000)
        ],

        # autoresume=True
    )

    trainer.fit()
