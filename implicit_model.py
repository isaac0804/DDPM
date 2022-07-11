import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

from composer import Trainer
from composer import ComposerModel
from composer.callbacks import LRMonitor
from composer.loggers import FileLogger, LogLevel, WandBLogger
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler

import math

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

class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, patch_size, expanding_ratio=4):
        super().__init__()
        self.cross_mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*expanding_ratio),
            nn.GELU(),
            nn.Linear(dim*expanding_ratio, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
    
    def forward(self, z, x):
        z = z + x
        attn, attn_weight = self.mha(z, z, z)
        z = self.norm1(attn + z)
        z = self.norm2(self.mlp(z) + z)

        return z

def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0, sampling=False):
    """ Anderson acceleration for fixed point iteration. """
    # bsz, d, H, W = x0.shape
    # dim = d * H * W
    bsz, d, n = x0.shape
    dim = d*n
    X = torch.zeros(bsz, m, dim, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, dim, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.contiguous().view(bsz, -1), f(x0).contiguous().view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).contiguous().view(bsz, -1)
    
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
        F[:,k%m] = f(X[:,k%m].view_as(x0)).contiguous().view(bsz, -1)
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
            z = self.solver(lambda z: self.f(z, x), torch.randn_like(x), sampling=sampling, **self.kwargs)
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        x = x.transpose(0, 1)
        return self.dropout(x)

class TransformerImplicitModel(nn.Module):
    def __init__(self, channels, latent_dim, num_heads, patch_size):
        # Patchify
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.patchify = nn.Conv2d(channels, latent_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pe = PositionalEncoding(latent_dim)

        self.class_embedding = nn.Embedding(10, latent_dim, device=self.device)

        f = TransformerLayer(latent_dim, num_heads, patch_size)
        self.deq = DEQFixedPoint(f, anderson, max_iter=50, tol=1e-2)

        self.fold = nn.Fold((32, 32), (2, 2), stride=(2, 2))
        self.conv = nn.Sequential(
            nn.Conv2d(latent_dim//patch_size**2, channels*4, kernel_size=3, padding="same", padding_mode="replicate"),
            nn.BatchNorm2d(channels*4),
            nn.GELU(),
            nn.Conv2d(channels*4, channels*2, kernel_size=3, padding="same", padding_mode="replicate"),
            nn.BatchNorm2d(channels*2),
            nn.GELU(),
            nn.Conv2d(channels*2, channels, kernel_size=1)
        )
        self.classify = nn.Linear(latent_dim, 10)

    def forward(self, inputs, labels):

        b, c, h, w = inputs.shape

        labels = self.class_embedding(labels)
        inputs = self.patchify(inputs).flatten(-2)
        inputs = torch.cat([inputs, labels[:, :, None]], dim=-1).transpose(-1, -2).contiguous()
        inputs = self.pe(inputs)

        inputs = self.deq(inputs).transpose(-1, -2)

        pred_labels = self.classify(inputs[:, :, -1])
        inputs = inputs[:, :, :-1]
        inputs = self.fold(inputs)
        inputs = self.conv(inputs)

        return inputs, F.softmax(pred_labels, dim=-1)
    
    def sampling(self, noise, labels_idx=None):

        b, c, h, w = noise.shape

        if not labels_idx:
            labels_idx = torch.randint(0, self.class_embedding.num_embeddings-1, (b,), device=self.device).long()
        labels = self.class_embedding(labels_idx)
        noise = self.patchify(noise).flatten(-2)
        noise = torch.cat([noise, labels[:, :, None]], dim=-1).transpose(-1, -2).contiguous()

        noise = self.deq(noise, sampling=True)
        noise = torch.cat(noise, dim=0).transpose(-1, -2)

        noise = noise[:, :, :-1]
        noise = self.fold(noise)
        noise = self.conv(noise)
        noise = noise.view(-1, b, c, h, w).permute(1, 0, 3, 4, 2)

        return noise, labels_idx

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
        images, labels = batch 
        noise = torch.randn_like(images)
        return self.model(noise, labels)

    def loss(self, output, batch):
        images, labels = batch
        gen_images, pred_labels = output
        recon_loss = self.criterion(gen_images, images)
        cls_loss = F.cross_entropy(pred_labels, labels)
        return recon_loss + cls_loss

    @torch.no_grad()
    def sample(self, num_samples=4):
        noise = torch.randn(num_samples, self.channels,
                            self.image_size, self.image_size, device=self.device)
        """
        noise = self.model.norm1(self.model.conv1(noise)).contiguous()
        noise = self.model.deq(noise, sampling=True)
        noise = torch.cat(noise, dim=0)
        noise = self.model.conv2(self.model.norm2(noise))
        # convert to [num_samples, batch_size, image_size, image_size, channels]
        noise = noise.view(-1, num_samples, self.channels, self.image_size, self.image_size).permute(1, 0, 3, 4, 2) 
        """
        images, labels = self.model.sampling(noise)

        return images, labels

if __name__ == "__main__":

    batch_size = 64
    image_size = 32
    channels = 1
    latent_dim = 16
    inner_dim = 64
    num_groups = 8

    num_heads = 8 
    patch_size = 2

    initial_lr = 1e-3
    final_lr = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_dataset, _ = get_cifar10(gray_scale=(channels==1))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    # model = ResnetImplicitModel(channels=channels, latent_dim=latent_dim, inner_dim=inner_dim, num_groups=num_groups).to(device)
    model = TransformerImplicitModel(channels, latent_dim, num_heads, patch_size)
    model = DDIM(model, batch_size, image_size, channels, loss_type="huber")
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    run_name = "ddim-transformer-gray-l"

    print("# Parameters:", sum(p.numel() for p in model.parameters()))

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
