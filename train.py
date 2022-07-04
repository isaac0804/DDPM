import logging
import os
from matplotlib import animation, pyplot as plt
import numpy as np

import torch
from torch.optim import Adam

from composer import Trainer
from composer.loggers import FileLogger, LogLevel, WandBLogger
from composer.core.callback import Callback
from composer.callbacks import LRMonitor
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler

from data import get_cifar10
from model import UNet

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("composer").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    device = "gpu" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    timesteps = 200

    train_dataset, _ = get_cifar10()
    # train_dataset, test_dataset = get_cifar10()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, drop_last=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True, num_workers=2, drop_last=True)

    file_logger = FileLogger(
        filename="runs/{run_name}/logs-rank{rank}.txt",
        buffer_size=1,
        capture_stderr=False,
        capture_stdout=False,
        log_level=LogLevel.BATCH,
        log_interval=100,
        flush_interval=200
    )
    # tb_logger = TensorboardLogger(log_dir="runs", flush_interval=1000)
    wandb_logger = WandBLogger()

    model = UNet(
        dim=32, # Image size
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=12,
        use_convnext=True,
        convnext_mult=2,
        batch_size=batch_size,
        timesteps=200,
        loss_type="huber")
    optimizer = Adam(model.parameters(), lr=1e-3)

    lr_scheduler = CosineAnnealingWithWarmupScheduler(t_warmup="5ep", alpha_f=1e-4)

    class SamplingCallback(Callback):
        def epoch_end(self, state, logger):

            fig = plt.figure()
            samples = state.model.sample(image_size=32)
            ims = []
            for i in range(model.timesteps):
                samples[i][0] = np.clip(samples[i][0] /2 + 0.5, 0, 1)
                im = plt.imshow(samples[i][0].transpose(1, 2, 0), animated=True)
                ims.append([im])
            animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
            folder = f"runs/{state.run_name}/samples"
            if os.path.exists(folder):
                animate.save(f"{folder}/sample-{state.timestamp.epoch}.gif")
            else:
                os.makedirs(folder)
                animate.save(f"{folder}/sample-{state.timestamp.epoch}.gif")
    
    run_name = "default-huber-warmup_cosine"

    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        max_duration="30ep",

        train_dataloader=train_loader,
        device=device,

        run_name=run_name,
        save_folder=f"runs/{run_name}/checkpoints",
        save_interval="5ep",

        schedulers=[
            lr_scheduler
        ],
        callbacks=[
            SamplingCallback(),
            LRMonitor()
        ],
        loggers=[
            file_logger,
            wandb_logger
        ]
    )

    trainer.fit()