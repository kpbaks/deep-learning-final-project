#!/usr/bin/env -S pixi run python3
# %%
import argparse
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import signal

import neptune
import torch

# import torchinfo
from loguru import logger

# import tqdm
from tqdm.rich import tqdm

try:
    from rich import pretty, print

    pretty.install()
except ImportError or ModuleNotFoundError:
    pass

from dataset import DrumsDataset
from model import Discriminator, Generator


@dataclass(frozen=True)
class Params:
    generator_lr: float
    discriminator_lr: float
    batch_size: int
    input_sz: int
    n_classes: int
    latent_sz: int
    leaky_relu_negative_slope: float
    num_epochs: int
    seed: int
    save_model_every: int

    def __post_init__(self) -> None:
        if self.generator_lr <= 0:
            raise ValueError(f'{self.generator_lr = } must be positive')
        if self.discriminator_lr <= 0:
            raise ValueError(f'{self.discriminator_lr = } must be positive')
        if self.batch_size <= 0:
            raise ValueError(f'{self.batch_size = } must be positive')
        if not is_power_of_2(self.batch_size):
            raise ValueError(f'{self.batch_size = } must be a power of 2')
        if self.input_sz <= 0:
            raise ValueError(f'{self.input_sz = } must be positive')
        if self.n_classes <= 0:
            raise ValueError(f'{self.n_classes = } must be positive')
        if self.latent_sz <= 0:
            raise ValueError(f'{self.latent_sz = } must be positive')
        if self.num_epochs <= 0:
            raise ValueError(f'{self.num_epochs = } must be positive')
        if self.seed <= 0:
            raise ValueError(f'{self.seed = } must be positive')
        if self.save_model_every <= 0:
            raise ValueError(f'{self.save_model_every = } must be positive')


def is_power_of_2(n: int) -> bool:
    """
    Check if a number is a power of 2.
    """
    return (n & (n - 1) == 0) and n != 0


def print_cuda_memory_usage() -> None:
    """Print the CUDA memory usage."""
    RESET: str = '\033[0m'
    GREEN: str = '\033[32m'
    YELLOW: str = '\033[33m'
    ORANGE: str = '\033[38;5;208m'
    RED: str = '\033[31m'

    def format_as_mibibytes(n: int) -> str:
        return f'{n / (1024 * 1024):.2f} MiB'

    for i in range(torch.cuda.device_count()):
        max_memory: int = torch.cuda.get_device_properties(i).total_memory
        allocated_memory: int = torch.cuda.memory_allocated(i)
        available_memory: int = max_memory - allocated_memory
        # percentage_available: float = available_memory / max_memory * 100.0
        percentage_used: float = allocated_memory / max_memory * 100.0
        if percentage_used > 90.0:
            color: str = RED
        elif percentage_used > 80.0:
            color = ORANGE
        elif percentage_used > 70.0:
            color = YELLOW
        else:
            color = GREEN
        logger.info(
            f'{color} CUDA device {i} available_memory/max_memory {format_as_mibibytes(available_memory)} / {format_as_mibibytes(max_memory)} {percentage_used = :.2f}%{RESET}'
        )


def train(
    g: Generator,
    d: Discriminator,
    device: torch.device,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    params: Params,
    run: neptune.Run,
) -> None:
    run['parameters'] = asdict(params)

    # lr = params.lr
    num_epochs = params.num_epochs

    g_optim = torch.optim.Adam(g.parameters(), lr=params.generator_lr, betas=(0.5, 0.999))
    d_optim = torch.optim.SGD(d.parameters(), lr=params.discriminator_lr, momentum=0.9)

    criterion = torch.nn.BCELoss()

    g.train()
    d.train()

    g_losses: list[float] = []
    d_losses: list[float] = []

    # Establish convention for real and fake labels during training
    real_label: float = 1.0
    fake_label: float = 0.0

    # Visualization of the generator progression
    _fixed_noise = torch.randn(64, params.latent_sz + params.n_classes, 1, 1, device=device)

    # TODO: maybe initialize weights here
    def save_snapshot_of_model_and_optimizer(
        model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int
    ) -> None:
        model_dir = Path.cwd() / 'models'
        model_dir.mkdir(exist_ok=True)
        # assert run._id is not None
        run_dir = model_dir / run['sys/id'].fetch() / f'epoch_{epoch + 1}'
        run_dir.mkdir(exist_ok=True)
        model_path = run_dir / f'{model.__class__.__name__}.pth'
        optimizer_path = run_dir / f'{optimizer.__class__.__name__}.pth'
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)
        logger.info(f'saved model at epoch {epoch + 1}')

    # Training is split up into two main parts. Part 1 updates the Discriminator and Part 2 updates the Generator.
    t_total = 0.0
    for epoch in range(num_epochs):

        def save_snapshot_of_both_models_and_optimizers() -> None:
            save_snapshot_of_model_and_optimizer(g, g_optim, epoch)
            save_snapshot_of_model_and_optimizer(d, d_optim, epoch)

        def ctrl_c_handler(signum: int, frame: object) -> None:
            logger.info('saving snapshot of models and optimizers')
            save_snapshot_of_both_models_and_optimizers()
            sys.exit(0)

        signal.signal(signal.SIGINT, ctrl_c_handler)

        t_start = time.time()

        logger.info(f'starting epoch {epoch + 1} / {num_epochs}')
        print_cuda_memory_usage()

        for i, (data, drum_type) in enumerate(tqdm(train_dataloader), 0):
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
                batch_size, params.latent_sz + params.n_classes, 1, 1, device=device
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

            # Run evaluation on validation set
            with torch.no_grad():
                d.eval()
                g.eval()
                # run on valudation set
                for i, (data, drum_type) in enumerate(val_dataloader, 0):
                    data = data.to(device)
                    batch_size = data.size(0)
                    label = torch.full((batch_size,), real_label, device=device)
                    # Forward pass with real data
                    output_real = d(data).view(-1)
                    # Loss on real data
                    d_x = output_real.mean().item()
                    run['validation/accuracy/Discriminator_accuracy_real'].append(d_x)

                    # Train Discriminator with all-fake batch
                    # Generate batch of latent vectors (latent vector size = 260)
                    noise = torch.randn(
                        batch_size, params.latent_sz + params.n_classes, 1, 1, device=device
                    )
                    # Generate fake data batch with Generator
                    fake_data = g(noise)
                    label.fill_(fake_label)

                    # Use discriminator to classify all-fake batch
                    output_fake = d(fake_data.detach()).view(-1)
                    d_g_z1 = output_fake.mean().item()
                    run['validation/accuracy/Discriminator_accuracy_fake'].append(d_g_z1)

                    d_acc = (d_x + d_g_z1) / 2
                    run['validation/accuracy/Discriminator_accuracy_total'].append(d_acc)

            # Output training stats
            if i % 5 == 0:
                print(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (
                        epoch,
                        num_epochs,
                        i,
                        len(train_dataloader),
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

        t_end = time.time()
        t_total += t_end - t_start
        logger.info(
            f'epoch {epoch + 1} / {num_epochs} took {t_end - t_start:.2f} seconds, total time: {t_total:.2f} seconds'
        )

        # Save model every N epochs
        if (epoch + 1) % params.save_model_every == 0:
            save_snapshot_of_both_models_and_optimizers()
            # save_snapshot_of_model_and_optimizer(g, g_optim, epoch)
            # save_snapshot_of_model_and_optimizer(d, d_optim, epoch)

            # model_dir = Path.cwd() / 'models'
            # model_dir.mkdir(exist_ok=True)
            # # assert run._id is not None
            # run_dir = model_dir / run['sys/id'].fetch()
            # run_dir.mkdir(exist_ok=True)
            # g_path = run_dir / f'g_{epoch + 1}.pth'
            # d_path = run_dir / f'd_{epoch + 1}.pth'
            # torch.save(g.state_dict(), g_path)
            # torch.save(d.state_dict(), d_path)
            # logger.info(f'saved model at epoch {epoch + 1}')

    run.stop()


def test(
    g: Generator,
    d: Discriminator,
    device: torch.device,
    test_dataloader: torch.utils.data.DataLoader,
    params: Params,
    run: neptune.Run,
) -> None:
    g.eval()
    d.eval()

    _criterion = torch.nn.BCELoss()

    # Establish convention for real and fake labels during training
    _real_label: float = 1.0
    _fake_label: float = 0.0

    with torch.no_grad():
        for i, (data, drum_type) in enumerate(tqdm(test_dataloader), 0):
            data = data.to(device)

    pass


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
    argv_parser.add_argument('--glr', type=float, default=0.0002, help='learning rate')
    argv_parser.add_argument(
        '--dlr', type=float, default=0.0002, help='discriminator learning rate'
    )
    argv_parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    argv_parser.add_argument(
        '--save-model-every', type=int, default=5, help='save model every N epochs'
    )
    argv_parser.add_argument(
        '--dataset-dir', type=str, help='path to dataset directory', default='~/datasets/drums'
    )
    args = argv_parser.parse_args()

    if args.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        raise ValueError(
            f'invalid log level {args.log_level = }, must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL'
        )
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    if args.epochs <= 0:
        raise ValueError(f'{args.epochs = } must be positive')

    if args.seed <= 0:
        raise ValueError(f'{args.seed = } must be positive')

    if args.glr <= 0:
        raise ValueError(f'{args.glr = } must be positive')

    if args.dlr <= 0:
        raise ValueError(f'{args.dlr = } must be positive')

    if args.batch_size <= 0:
        raise ValueError(f'{args.batch_size = } must be positive')

    if not is_power_of_2(args.batch_size):
        raise ValueError(f'{args.batch_size = } must be a power of 2')

    if args.save_model_every <= 0:
        raise ValueError(f'{args.save_model_every = } must be positive')

    if args.save_model_every > args.epochs:
        logger.warning(
            f'{args.save_model_every = } is greater than {args.epochs = }, no models will be saved'
        )

    dataset_dir: Path = Path(args.dataset_dir).expanduser()
    if not dataset_dir.exists():
        raise ValueError(f'{dataset_dir} does not exist')
    if not dataset_dir.is_dir():
        raise ValueError(f'{dataset_dir} is not a directory')

    params = Params(
        generator_lr=args.glr,
        discriminator_lr=args.dlr,
        batch_size=args.batch_size,
        input_sz=2 * 128 * 512,
        n_classes=4,
        latent_sz=256,
        leaky_relu_negative_slope=0.2,
        num_epochs=args.epochs,
        seed=args.seed,
        save_model_every=args.save_model_every,
    )

    print(f'{params = }')

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results
    logger.info(f'{torch.cuda.is_available() = }')
    device = select_cuda_device_by_memory() or torch.device('cpu')
    logger.info(f'{device = } {torch.cuda.device_count() = }')

    g = Generator(params.latent_sz, params.n_classes, params.leaky_relu_negative_slope).to(device)
    d = Discriminator(params.leaky_relu_negative_slope).to(device)

    logger.debug(f'{g = }')
    logger.debug(f'{d = }')

    # dataset_dir = Path.home() / 'datasets' / 'drums'
    train_dir = dataset_dir / 'train'
    assert train_dir.exists(), f'{train_dir} does not exist'
    assert train_dir.is_dir(), f'{train_dir} is not a directory'
    test_dir = dataset_dir / 'test'
    assert test_dir.exists(), f'{test_dir} does not exist'
    assert test_dir.is_dir(), f'{test_dir} is not a directory'
    val_dir = dataset_dir / 'val'
    assert val_dir.exists(), f'{val_dir} does not exist'
    assert val_dir.is_dir(), f'{val_dir} is not a directory'

    train_dataset = DrumsDataset(train_dir)
    test_dataset = DrumsDataset(test_dir)
    val_dataset = DrumsDataset(val_dir)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=2
    )
    _test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=params.batch_size, shuffle=True, num_workers=2
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=params.batch_size, shuffle=True, num_workers=2
    )

    print_cuda_memory_usage()

    run: neptune.Run = neptune.init_run(
        project='jenner/deep-learning-final-project',
        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNzNkMDgxNS1lOTliLTRjNWQtOGE5Mi1lMDI5NzRkMWFjN2MifQ==',
    )
    logger.info(f'{run._id = }')
    logger.info(f'{run._name = }')
    logger.info(f"{run['sys/id'].fetch() = }")

    logger.info('starting training')
    train(
        g=g,
        d=d,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        params=params,
        run=run,
    )

    logger.info('starting testing')
    test(
        g=g,
        d=d,
        device=device,
        test_dataloader=_test_dataloader,
        params=params,
        run=run,
    )

    return 0


if __name__ == '__main__':
    logger.debug(f'{sys.executable = }')
    main()

# %%
