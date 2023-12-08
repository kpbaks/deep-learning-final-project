#!/usr/bin/env -S pixi run python3
# %%
# import os
import time
import random
import sys
import argparse

# from torchvision import utils as vutils
# import sys
# from dataclasses import asdict, astuple, dataclass
from pathlib import Path

import neptune
import torch

# import torchinfo
from loguru import logger

try:
    from rich import pretty, print

    pretty.install()
except ImportError or ModuleNotFoundError:
    pass

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
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    params: dict,
    # lr: float, #Moved to neptune params
    # num_epochs: int, #Moved to neptune params
) -> None:
    run = neptune.init_run(
        project='jenner/deep-learning-final-project',
        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNzNkMDgxNS1lOTliLTRjNWQtOGE5Mi1lMDI5NzRkMWFjN2MifQ==',
    )
    run['parameters'] = params

    lr = params['lr']
    num_epochs = params['num_epochs']

    if lr <= 0:
        raise ValueError(f'{lr = } must be positive')
    if num_epochs <= 0:
        raise ValueError(f'{num_epochs = } must be positive')

    # for epoch in range(10):
    #    run['train/loss'].append(0.9**epoch)

    # run['eval/f1_score'] = 0.66

    g_optim = torch.optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optim = torch.optim.SGD(d.parameters(), lr=lr, momentum=0.9)

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
    _fixed_noise = torch.randn(64, params['latent_sz'] + params['n_classes'], 1, 1, device=device)

    # TODO: maybe initialize weights here
    # TODO: maybe use custom tqdm progress bar

    # Training is split up into two main parts. Part 1 updates the Discriminator and Part 2 updates the Generator.
    for epoch in range(num_epochs):
        t_start = time.time()
        logger.info(f'starting epoch {epoch + 1} / {num_epochs}')
        for i, (data, drum_type) in enumerate(dataloader, 0):
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
            run['train/error/Discriminator_loss_real'].append(d_err_real)
            # gradients for D in backward pass
            d_err_real.backward()
            d_x = output_real.mean().item()
            run['train/accuracy/Discriminator_accuracy_real'].append(d_x)

            # Train Discriminator with all-fake batch
            # Generate batch of latent vectors (latent vector size = 260)
            noise = torch.randn(
                batch_size, params['latent_sz'] + params['n_classes'], 1, 1, device=device
            )
            # Generate fake data batch with Generator
            fake_data = g(noise)
            label.fill_(fake_label)

            # Use discriminator to classify all-fake batch
            output = d(fake_data.detach()).view(-1)
            d_err_fake = criterion(output, label)
            run['train/error/Discriminator_loss_fake'].append(d_err_fake)

            # Calculate gardients for D in backward pass
            d_err_fake.backward()
            d_g_z1 = output.mean().item()
            run['train/accuracy/Discriminator_accuracy_fake'].append(d_g_z1)

            d_err = d_err_real + d_err_fake
            d_acc = (d_x + d_g_z1) / 2
            run['train/error/Discriminator_loss_total'].append(d_err)
            run['train/accuracy/Discriminator_accuracy_total'].append(d_acc)
            d_optim.step()

            # Update Generator -> maximize log(D(G(z)))
            g.zero_grad()
            label.fill_(real_label)  # fake labels are real for the generator

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = d(fake_data).view(-1)

            # Calculate generator loss based on new output from discriminator

            g_err = criterion(output, label)
            run['train/error/Generator_error'].append(g_err)
            # Backwardspass
            g_err.backward()
            d_g_z2 = output.mean().item()
            run['train/accuracy/Generator_accuracy'].append(d_g_z2)
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
        t_end = time.time()
        logger.info(f'epoch {epoch + 1} / {num_epochs} took {t_end - t_start:.2f} seconds')

    run.stop()


def select_cuda_device_by_memory() -> torch.device | None:
    """Select the CUDA device with the most available memory."""
    # Get all CUDA devices
    cuda_devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    if len(cuda_devices) == 0:
        return None

    device: torch.device | None = None
    max_available_memory: int = 0

    # Get the memory usage of each device
    for cuda_device in cuda_devices:
        total_memory: int = torch.cuda.get_device_properties(cuda_device).total_memory
        allocated_memory: int = torch.cuda.memory_allocated(cuda_device)
        available_memory: int = total_memory - allocated_memory
        logger.debug(f'{cuda_device = } {available_memory = } {total_memory = }')
        if available_memory > max_available_memory:
            max_available_memory = available_memory
            device = cuda_device

    assert device is not None
    return device


def main() -> int:
    argv_parser = argparse.ArgumentParser()
    argv_parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    argv_parser.add_argument('--seed', type=int, default=1234, help='random seed')
    argv_parser.add_argument('--log-level', type=str, default='INFO', help='log level')
    argv_parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    argv_parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    args = argv_parser.parse_args()

    if args.epochs <= 0:
        raise ValueError(f'{args.epochs = } must be positive')

    if args.seed <= 0:
        raise ValueError(f'{args.seed = } must be positive')

    if args.lr <= 0:
        raise ValueError(f'{args.lr = } must be positive')

    if args.batch_size <= 0:
        raise ValueError(f'{args.batch_size = } must be positive')

    if not is_power_of_2(args.batch_size):
        raise ValueError(f'{args.batch_size = } must be a power of 2')

    params = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'input_sz': 2 * 128 * 512,
        'n_classes': 4,
        'latent_sz': 256,
        'leaky_relu_negative_slope': 0.2,
        'num_epochs': args.epochs,
        'seed': args.seed,
    }

    print(f'{params = }')

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results
    logger.info(f'{torch.cuda.is_available() = }')
    device = select_cuda_device_by_memory() or torch.device('cpu')
    logger.info(f'{device = }')
    npgu: int = torch.cuda.device_count()
    logger.info(f'{npgu = } {device = }')

    g = Generator(params['latent_sz'], params['n_classes'], params['leaky_relu_negative_slope']).to(
        device
    )

    d = Discriminator(params['leaky_relu_negative_slope']).to(device)

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

    # generator_stats = torchinfo.summary(g, input_size=(batch_size, 268, 1, 1))
    # discriminator_stats = torchinfo.summary(d, input_size=(batch_size, 2, 128, 512))
    # logger.info(f'{generator_stats = }')
    # logger.info(f'{discriminator_stats = }')

    # sizeof_dataset_sample: int = dataset[0].element_size() * dataset[0].numel()
    # sizeof_batch: int = sizeof_dataset_sample * batch_size
    # sizeof_dataset: int = sizeof_dataset_sample * len(dataset)
    # logger.info(f'{sizeof_dataset_sample = } B {sizeof_batch = } B {sizeof_dataset = } B')

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=params['batch_size'], shuffle=True, num_workers=2
    )

    train(g, d, device, dataloader, params)
    return 0


if __name__ == '__main__':
    logger.debug(f'{sys.executable = }')
    main()
