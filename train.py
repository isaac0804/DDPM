import logging
from cv2 import log

import torch
from torch.optim import Adam
from torchvision import datasets, transforms

from composer import Trainer
from composer.algorithms import ChannelsLast
from composer.loggers import FileLogger, LogLevel, TensorboardLogger
from composer.callbacks import CheckpointSaver

from data import get_cifar10
from model import UNet

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    device = "gpu" if torch.cuda.is_available() else "cpu"
    batch_size = 32
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
        log_interval=10,
        flush_interval=20
    )
    # tb_logger = TensorboardLogger(log_dir="runs", flush_interval=10)

    model = UNet(
        dim=32, # Image size
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
        batch_size=batch_size,
        timesteps=200,
        loss_type="huber")
    optimizer = Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        # eval_dataloader=test_loader,
        optimizers=optimizer,
        device=device,
        max_duration="2ep",
        save_folder="runs/{run_name}/checkpoints",
        save_interval="1ep",
        algorithms=[
            ChannelsLast(),
        ],
        loggers=[
            file_logger
        ],
    )

    trainer.fit()