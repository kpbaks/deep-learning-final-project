#!/usr/bin/env -S pixi run python3
# %%
# import os
import random
import sys

# from torchvision import utils as vutils
# import sys
# from dataclasses import asdict, astuple, dataclass
from pathlib import Path

import torch

# import torchinfo
from loguru import logger

from dataset import DrumsDataset
from model import Discriminator, Generator

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


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

    _img_list = []
    iters = 0
    # Visualization of the generator progression
    _fixed_noise = torch.randn(64, 256, 1, 1, device=device)

    # TODO: maybe initialize weights here
    # TODO: maybe use custom tqdm progress bar

    # Training is split up into two main parts. Part 1 updates the Discriminator and Part 2 updates the Generator.
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1} / {num_epochs + 1}')
        for i, data in enumerate(dataloader, 0):
            d.zero_grad()

            data = data.to(device)
            batch_size = data.size(0)

            real_data = data.to(device)
            batch_size = real_data.size(0)
            label = torch.full((batch_size,), real_label, device=device)
            # Forward pass with real data
            output_real = d(real_data).view(-1)
            assert output_real.shape == (batch_size,), f'{output_real.shape = }'

            # Loss on real data
            d_err_real = criterion(output_real, label)

            # gradients for D in backward pass
            d_err_real.backward()
            d_x = output_real.mean().item()

            # Train Discriminator with all-fake batch
            # Generate batch of latent vectors (latent vector size = 256)
            noise = torch.randn(batch_size, 268, 1, 1, device=device)
            # Generate fake data batch with Generator
            fake_data = g(noise)
            label.fill_(fake_label)

            # Use discriminator to classify all-fake batch
            output = d(fake_data.detach()).view(-1)
            d_err_fake = criterion(output, label)

            # Calculate gardients for D in backward pass
            d_err_fake.backward()
            d_g_z1 = output.mean().item()

            d_err = d_err_real + d_err_fake
            d_optim.step()

            # Update Generator -> maximize log(D(G(z)))
            g.zero_grad()
            label.fill_(real_label)  # fake labels are real for the generator

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = d(fake_data).view(-1)

            # Calculate generator loss based on new output from discriminator

            g_err = criterion(output, label)
            # Backwardspass
            g_err.backward()
            d_g_z2 = output.mean().item()
            # Update Generator
            g_optim.step()

            # Output training stats
            if i % 5 == 0:
                print(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (
                        epoch,
                        num_epochs,
                        i,
                        len(dataloader),
                        d_err.item(),
                        g_err.item(),
                        d_x,
                        d_g_z1,
                        d_g_z2,
                    )
                )

            # Save Losses for plotting later
            g_losses.append(g_err.item())
            d_losses.append(d_err.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            # if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            # with torch.no_grad():
            # fake = g(fixed_noise).detach().cpu()
            # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1


def main() -> int:
    seed: int = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results
    logger.info(f'{torch.cuda.is_available() = }')
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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

    # generator_stats = torchinfo.summary(g, input_size=(batch_size, 268, 1, 1))
    # discriminator_stats = torchinfo.summary(d, input_size=(batch_size, 2, 128, 512))
    # logger.info(f'{generator_stats = }')
    # logger.info(f'{discriminator_stats = }')

    sizeof_dataset_sample: int = dataset[0].element_size() * dataset[0].numel()
    sizeof_batch: int = sizeof_dataset_sample * batch_size
    sizeof_dataset: int = sizeof_dataset_sample * len(dataset)
    logger.info(f'{sizeof_dataset_sample = } B {sizeof_batch = } B {sizeof_dataset = } B')

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    # logger.info(f'{generator_stats.total_params = }')

    train(g, d, dataloader, lr=0.0002, num_epochs=10)

    return 0


if __name__ == '__main__':
    logger.debug(f'{sys.executable = }')
    main()
    # sys.exit()
    # sys.exit(main(len(sys.argv), sys.argv))
    # pass

# %%
