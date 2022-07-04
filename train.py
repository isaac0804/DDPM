import logging

import torch
from composer import Trainer
from composer.callbacks import LRMonitor
from composer.loggers import FileLogger, LogLevel, WandBLogger
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler
from torch.optim import Adam

from callbacks import SamplingCallback
from data import get_cifar10
from model import DDPM, UNet

if __name__ == "__main__":

    # Logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("composer").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # Hyperparameters
    device = "gpu" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    image_size = 32
    channels = 1
    timesteps = 200
    initial_lr = 1e-3
    final_lr = 1e-4

    # Data
    train_dataset, _ = get_cifar10(gray_scale=(channels==1))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Model
    unet = UNet(
        dim=image_size,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=channels,
        with_time_emb=True,
        resnet_block_groups=12,
        use_convnext=True,
        convnext_mult=2
    )
    model = DDPM(
        model=unet,
        batch_size=batch_size,
        image_size=image_size,
        channels=channels,
        timesteps=timesteps,
        loss_type="huber"
    )
    optimizer = Adam(model.parameters(), lr=initial_lr)

    run_name = "default-huber-gray"

    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        max_duration="30ep",

        train_dataloader=train_loader,
        device=device,

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
            SamplingCallback(),
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
                log_level=LogLevel.BATCH,
                log_interval=100,
                flush_interval=200
            )
            # TensorboardLogger(log_dir="runs", flush_interval=1000)
        ]
    )

    trainer.fit()
