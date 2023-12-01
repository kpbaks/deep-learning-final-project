#!/usr/bin/env -S pixi run python3
# %%
# import os
import random
# import sys
# from dataclasses import asdict, astuple, dataclass

import torch
from loguru import logger
from torch import nn as nn

# from pathlib import Path
# import argparse
# import time
from model import Discriminator, Generator

# from torch import Tensor
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transformsp


# @dataclass
# class Config:


def train(
    g: Generator,
    d: Discriminator,
    dataloader: torch.utils.data.Dataloader,
    lr: float,
    num_epochs: int,
) -> None:
    assert lr > 0, f'{lr = } must be positive'
    assert num_epochs > 0, f'{num_epochs = } must be positive'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    g_optim = torch.optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optim = torch.optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    g.train()
    d.train()

    g_losses: list[float] = []
    d_losses: list[float] = []

    # TODO: maybe initialize weights here
    # TODO: maybe use custom tqdm progress bar
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            d.zero_grad()

            data = data.tlisto(device)
            batch_size = data.size(0)

            # Train discriminator with real data
            d_real_decision = d(data).view(-1)
            assert d_real_decision.shape == (batch_size,), f'{d_real_decision.shape = }'


def main() -> int:
    seed: int = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    npgu: int = torch.cuda.device_count()
    logger.info(f'{npgu = }')
    logger.info(f'{device = }')

    latent_size: int = 256
    pitch_conditioning_size: int = 3 * 4

    g = Generator(latent_size, pitch_conditioning_size, leaky_relu_negative_slope=0.2).to(device)

    d = Discriminator(leaky_relu_negative_slope=0.2).to(device)

    if (device.type == 'cuda') and (npgu > 1):
        logger.info(f'Using {npgu} GPUs')
        g = torch.nn.DataParallel(g, list(range(npgu)))
        d = torch.nn.DataParallel(d, list(range(npgu)))

    logger.debug(f'{g = }')
    logger.debug(f'{d = }')

    return 0


if __name__ == '__main__':
    # sys.exit(main(len(sys.argv), sys.argv))
    main()
    # sys.exit(main(len(sys.argv), sys.argv))
