#!/usr/bin/env -S pixi run python3
# %%
# import os
import random
import sys

# import sys
# from dataclasses import asdict, astuple, dataclass
from pathlib import Path

import torch
import torchinfo
from loguru import logger

from dataset import DrumsDataset
from model import Discriminator, Generator


def is_power_of_2(n: int) -> bool:
    """
    Check if a number is a power of 2.
    """
    return (n & (n - 1) == 0) and n != 0


def train(
    g: Generator,
    d: Discriminator,
    dataloader: torch.utils.data.DataLoader,
    lr: float,
    num_epochs: int,
) -> None:
    assert lr > 0, f'{lr = } must be positive'
    assert num_epochs > 0, f'{num_epochs = } must be positive'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    g_optim = torch.optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optim = torch.optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion = torch.nn.BCELoss()

    g.train()
    d.train()

    g_losses: list[float] = []
    d_losses: list[float] = []

    # Establish convention for real and fake labels during training
    real_label: float = 1.0
    fake_label: float = 0.0

    # TODO: maybe initialize weights here
    # TODO: maybe use custom tqdm progress bar

    # Training is split up into two main parts. Part 1 updates the Discriminator and Part 2 updates the Generator.
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            d.zero_grad()

            data = data.to(device)
            batch_size = data.size(0)

            # Train discriminator with real data
            d_real_decision = d(data).view(-1)
            assert d_real_decision.shape == (batch_size,), f'{d_real_decision.shape = }'


def main() -> int:
    seed: int = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results
    logger.info(f'{torch.cuda.is_available() = }')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    npgu: int = torch.cuda.device_count()
    logger.info(f'{npgu = } {device = }')

    latent_size: int = 256
    pitch_conditioning_size: int = 3 * 4

    g = Generator(latent_size, pitch_conditioning_size, leaky_relu_negative_slope=0.2).to(device)

    d = Discriminator(leaky_relu_negative_slope=0.2).to(device)

    # if (device.type == 'cuda') and (npgu > 1):
    #     logger.info(f'Using {npgu} GPUs')
    #     g = torch.nn.DataParallel(g, list(range(npgu)))
    #     d = torch.nn.DataParallel(d, list(range(npgu)))

    logger.debug(f'{g = }')
    logger.debug(f'{d = }')

    dataset_dir = Path.home() / 'datasets' / 'classic_clean'
    dataset = DrumsDataset(dataset_dir)

    # Estimate the highest batch size that fits in memory, based on the size of the dataset

    max_cuda_memory: int = torch.cuda.get_device_properties(device).total_memory
    logger.info(f'{max_cuda_memory = }')
    available_cuda_memory: int = max_cuda_memory - torch.cuda.memory_allocated(device)
    logger.info(f'{available_cuda_memory = }')
    cuda_memory_utilization_percentage: float = available_cuda_memory / max_cuda_memory * 100.0
    logger.info(f'{cuda_memory_utilization_percentage = }%')

    batch_size = 32
    assert is_power_of_2(batch_size), f'{batch_size = } must be a power of 2'

    generator_stats = torchinfo.summary(g, input_size=(batch_size, 268, 1, 1))
    discriminator_stats = torchinfo.summary(d, input_size=(batch_size, 2, 128, 512))
    logger.info(f'{generator_stats = }')
    logger.info(f'{discriminator_stats = }')

    sizeof_dataset_sample: int = dataset[0].element_size() * dataset[0].numel()
    sizeof_batch: int = sizeof_dataset_sample * batch_size
    sizeof_dataset: int = sizeof_dataset_sample * len(dataset)
    logger.info(f'{sizeof_dataset_sample = } B {sizeof_batch = } B {sizeof_dataset = } B')

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    logger.info(f'{generator_stats.total_params = }')

    train(g, d, dataloader, lr=0.0002, num_epochs=10)

    return 0


if __name__ == '__main__':
    logger.debug(f'{sys.executable = }')
    main()
    # sys.exit()
    # sys.exit(main(len(sys.argv), sys.argv))
    # pass
