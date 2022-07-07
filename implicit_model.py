import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from data import get_cifar10

# Implicit Layer Implementation

class ResnetImplicitLayer(nn.Module):
    def __init__(self, channels, inner_channels, kernel_size=3, num_groups=8):
        super().__init__()
        self.channels = channels
        self.inner_channels = inner_channels

        self.conv1 = nn.Conv2d(channels, inner_channels, kernel_size, padding=kernel_size//2, groups=num_groups, bias=False)
        self.conv2 = nn.Conv2d(inner_channels, channels, kernel_size, padding=kernel_size//2, groups=num_groups, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.norm3 = nn.GroupNorm(num_groups, channels)

    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(self.conv2(y) + x)))

def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0):
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
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:,k%m].view_as(x0), res

class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self, x):
        with torch.no_grad():
            z, self.foward_res = self.solver(lambda z: self.f(z, x), torch.zeros_like(x), **self.kwargs)
        z = self.f(z, x)

        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)
        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y: autograd.grad(f0, z0, grad_outputs=y, retain_graph=True)[0] + grad, grad, **self.kwargs)
            return g
        z.register_hook(backward_hook)
        return z

class ResnetImplicitModel(nn.Module):
    def __init__(self, channels, inner_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, inner_channels, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(inner_channels)

        f = ResnetImplicitLayer(inner_channels, 64, kernel_size=3, num_groups=8)
        self.deq = DEQFixedPoint(f, anderson, max_iter=50, tol=1e-2)

        self.norm2 = nn.BatchNorm2d(inner_channels)
        self.conv2 = nn.Conv2d(inner_channels, channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        x = self.norm1(self.conv1(x)).contiguous()
        x = self.norm2(self.deq(x))
        x = self.conv2(x)
        return x

"""
class DDPM_Implicit(ComposerModel):
    def __init__(self, model, batch_size, image_size, channels, timesteps, loss_type="huber", betas_type="linear"):
        super().__init__()
        self.model = model
        self.noise = None
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
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

    def forward(self, x):

        x, _ = x  # Excluding labels

        # time = torch.randint(
        #     0, self.timesteps, (self.batch_size,), device=self.device).long()
        # self.noise = torch.randn_like(x)
        # x = q_sample(x, t=time, betas=self.betas, noise=self.noise)
        noise = torch.randn_like(x)

        return self.model(noise, x)

    def loss(self, predicted_noise, batch):
        loss = self.criterion(predicted_noise, self.noise)
        return loss

    @torch.no_grad()
    def sample(self, show_progress=False):
        image = torch.randn(self.batch_size, self.channels,
                            self.image_size, self.image_size, device=self.device)
        images = []

        if not show_progress:
            iterator = reversed(range(0, self.timesteps))
        else:
            iterator = tqdm(reversed(range(0, self.timesteps)),
                            desc='sampling loop time step', total=self.timesteps)

        for i in iterator:
            # p sample
            t = torch.full((self.batch_size,), i,
                           dtype=torch.long, device=self.device)
            betas = self.betas
            alphas = get_alphas(self.betas)
            betas_t = extract(betas, t, image.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(
                get_sqrt_one_minus_alphas_cumprod(alphas), t, image.shape
            )
            sqrt_recip_alphas_t = extract(
                get_sqrt_recip_alphas(alphas), t, image.shape)

            # Equation 11 in the paper
            # Use our model (noise predictor) to predict the mean
            model_mean = sqrt_recip_alphas_t * (
                image - betas_t *
                self.model(image, t) / sqrt_one_minus_alphas_cumprod_t
            )

            if i == 0:
                image = model_mean
            else:
                posterior_variance_t = extract(
                    get_posterior_variance(alphas, betas), t, image.shape)
                noise = torch.randn_like(image)
                # Algorithm 2 line 4:
                image = model_mean + torch.sqrt(posterior_variance_t) * noise
            images.append(image.cpu().numpy())

        return images
"""

if __name__ == "__main__":

    epochs = 30
    batch_size = 64
    channels = 1
    inner_channels = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, _ = get_cifar10(gray_scale=(channels==1))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    model = ResnetImplicitModel(channels=channels, inner_channels=inner_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("# Parameters: ", sum(p.numel() for p in model.parameters()))

    writer = SummaryWriter(log_dir="./runs/cifar10_resnet_implicit")

    for epoch in range(epochs):
        total_loss = 0.
        for i, (images, _) in tqdm(enumerate(train_loader, 0), total=len(train_loader), ncols=100, desc="Training Progress"):
            images = images.to(device)
            noise = torch.randn_like(images, device=device)

            gen = model(noise)
            loss = F.huber_loss(gen, images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            it = len(train_loader) * epoch + i
            writer.add_scalar("Loss", loss.item(), it)
            total_loss += loss.item()
            
        print(f"Epoch   : {epoch+1}")
        print(f"Loss    : {total_loss/len(train_loader):4.9f}")

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch-{epoch+1}.pt")