#!/usr/bin/env -S pixi run python3
# %%
import argparse
import random
import sys

# import time
from dataclasses import asdict, dataclass
from pathlib import Path
import signal

import neptune
import torch
import matplotlib.pyplot as plt

# import torchinfo
from loguru import logger

# import tqdm
# from tqdm.rich import tqdm as rtqdm, trange as rtrange
from tqdm import tqdm, trange
import pretty_errors

from stft import spectrum_2_audio  # noqa

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
    train_generator_every: int

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
        if self.train_generator_every <= 0:
            raise ValueError(f'{self.train_generator_every = } must be positive')


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


def weights_init(m: torch.nn.Module, stddev: float = 0.02) -> None:
    """
    Initialize the weights of a module.
    Taken from: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#weight-initialization
    """
    classname: str = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, stddev)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, stddev)
        torch.nn.init.constant_(m.bias.data, 0)


def generate_noise(
    batch_size: int,
    latent_sz: int,
    n_classes: int,
    drum_types: list[str],
    device: torch.device,
) -> torch.Tensor:
    # Generate batch of latent vectors (latent vector size = 260)
    noise = torch.randn(batch_size, latent_sz, 1, 1)
    onehot_encoded_labels = torch.cat(
        [DrumsDataset.onehot_encode_label(drum_type) for drum_type in drum_types], dim=0
    )
    onehot_encoded_labels = onehot_encoded_labels.reshape(batch_size, n_classes, 1, 1)
    assert onehot_encoded_labels.shape == (
        batch_size,
        n_classes,
        1,
        1,
    ), f'{onehot_encoded_labels.shape = }'
    noise = torch.cat((noise, onehot_encoded_labels), dim=1).to(device)
    assert noise.shape == (
        batch_size,
        latent_sz + n_classes,
        1,
        1,
    ), f'{noise.shape = }'

    return noise


def train(
    g: Generator,
    d: Discriminator,
    device: torch.device,
    train_dataloader: torch.utils.data.DataLoader,
    params: Params,
    run: neptune.Run,
) -> None:
    run['parameters'] = asdict(params)

    # lr = params.lr
    num_epochs = params.num_epochs

    g_optim = torch.optim.Adam(g.parameters(), lr=params.generator_lr, betas=(0.5, 0.999))
    d_optim = torch.optim.Adam(d.parameters(), lr=params.discriminator_lr, betas=(0.5, 0.999))

    # TODO: try different loss functions
    # criterion = torch.nn.BCELoss()  # Binary Cross Entropy Loss

    # Initialize weights
    g.apply(weights_init)
    d.apply(weights_init)

    g.train()
    d.train()

    def save_snapshot_of_model_and_optimizer(
        model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int
    ) -> None:
        model_dir = Path.cwd() / 'models'
        model_dir.mkdir(exist_ok=True, parents=True)
        # assert run._id is not None
        run_dir = model_dir / run['sys/id'].fetch() / f'epoch_{epoch + 1}'
        run_dir.mkdir(exist_ok=True, parents=True)
        model_path = run_dir / f'{model.__class__.__name__.lower()}.pth'
        optimizer_path = run_dir / f'{optimizer.__class__.__name__.lower()}.pth'
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)
        logger.info(f'saved model at epoch {epoch + 1}')

    # Training is split up into two main parts. Part 1 updates the Discriminator and Part 2 updates the Generator.
    for epoch in trange(num_epochs, leave=False, colour='blue'):
        g.train()
        d.train()

        def save_snapshot_of_both_models_and_optimizers() -> None:
            save_snapshot_of_model_and_optimizer(g, g_optim, epoch)
            save_snapshot_of_model_and_optimizer(d, d_optim, epoch)

        def ctrl_c_handler(signum: int, frame: object) -> None:
            logger.info('saving snapshot of models and optimizers')
            save_snapshot_of_both_models_and_optimizers()
            sys.exit(0)

        def save_sample_picture(data: torch.Tensor, drum_types) -> None:
            with torch.no_grad():
                g.eval()
                # print(f"{data.shape = }")
                # print(f"{drum_types = }")
                latent_vec = generate_noise(
                    1, params.latent_sz, params.n_classes, [drum_types[0]], device
                )
                assert len(latent_vec.shape) == 4
                img = g(latent_vec).detach()[0]
                img = img.cpu().numpy()
                assert len(img.shape) == 3
                mag = img[0]
                assert len(mag.shape) == 2

                fig, axis = plt.subplots(2, 1, figsize=(8, 6))
                axis[0].set_title(f'epoch: {epoch + 1} drum type: {drum_types[0]}')
                im = axis[0].imshow(mag, origin='lower')
                fig.colorbar(im, ax=axis[0])

                audio = spectrum_2_audio(torch.from_numpy(img), 44100.0)
                axis[1].plot(audio)
                axis[1].set_title('audio')
                # Show colorbar

                run['gen/images'].append(fig)

        signal.signal(signal.SIGINT, ctrl_c_handler)

        progress_bar = tqdm(
            train_dataloader,
            desc=f'epoch {epoch + 1} / {num_epochs}',
            leave=False,
            colour='green',
        )
        for i, (data, drum_types) in enumerate(progress_bar, 0):
            d_optim.zero_grad()
            real_data = data.to(device)
            batch_size: int = real_data.size(0)

            noise = generate_noise(
                batch_size, params.latent_sz, params.n_classes, drum_types, device
            )
            fake_data = g(noise).detach()

            d_loss = -torch.mean(d(real_data)) + torch.mean(d(fake_data))
            d_loss.backward()
            d_optim.step()
            # grad_norm = torch.nn.utils.clip_grad_norm_(d.parameters(), max_norm=1e3, norm_type=2)
            # print("Discriminator gradient norm:", grad_norm)
            run['train/error/discriminator'].append(d_loss.item())

            # Clip weights of discriminator
            for p in d.parameters():
                p.data.clamp_(-0.01, 0.01)

            # Train Generator every N_critic iterations
            if i % params.train_generator_every == 0:
                g_optim.zero_grad()

                fake_data = g(noise)
                g_loss = -torch.mean(d(fake_data))
                g_loss.backward()
                g_optim.step()
                run['train/error/generator'].append(g_loss.item())

        # # Run evaluation on validation set
        # with torch.no_grad():
        #     d.eval()
        #     g.eval()
        #     # run on valudation set
        #     for i, (data, drum_types) in enumerate(val_dataloader, 0):
        #         real_data = data.to(device)
        #         batch_size = data.size(0)
        #         noise = generate_noise(
        #             batch_size, params.latent_sz, params.n_classes, drum_types, device
        #         )
        #         fake_data = g(noise)

        #         d_loss = -torch.mean(d(real_data)) + torch.mean(d(fake_data))
        #         run['val/error/discriminator'].append(d_loss.item())

        #         g_loss = -torch.mean(d(fake_data))
        #         run['val/error/generator'].append(g_loss.item())

        # Save model every N epochs
        if (epoch + 1) % params.save_model_every == 0:
            # idx = random.randint(0, len(train_dataloader) - 1)

            data, drum_types = next(iter(train_dataloader))
            # print(f'{drum_types = }')
            save_sample_picture(data, ['snare'])
            save_snapshot_of_both_models_and_optimizers()


# def test(
#     g: Generator,
#     d: Discriminator,
#     device: torch.device,
#     test_dataloader: torch.utils.data.DataLoader,
#     params: Params,
#     run: neptune.Run,
# ) -> None:
#     with torch.no_grad():
#         d.eval()
#         g.eval()
#         # run on valudation set
#         for i, (data, drum_types) in enumerate(test_dataloader, 0):
#             real_data = data.to(device)
#             batch_size = data.size(0)
#             noise = generate_noise(
#                 batch_size, params.latent_sz, params.n_classes, drum_types, device
#             )
#             fake_data = g(noise)

#             d_loss = -torch.mean(d(real_data)) + torch.mean(d(fake_data))
#             run['test/error/discriminator'].append(d_loss.item())

#             g_loss = -torch.mean(d(fake_data))
#             run['test/error/generator'].append(g_loss.item())


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
    argv_parser.add_argument('--seed', type=int, required=False, help='random seed')
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
        '--dataset-dir',
        type=str,
        help='path to dataset directory',
        default='~/datasets/drums',
    )
    argv_parser.add_argument('--name', type=str, help='name of the run', required=False)
    argv_parser.add_argument(
        '--train-generator-every',
        type=int,
        default=5,
        help='train generator every N iterations',
    )
    args = argv_parser.parse_args()

    if args.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        raise ValueError(
            f'invalid log level {args.log_level = }, must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL'
        )
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    logger.debug(f'{args = }')
    if args.epochs <= 0:
        raise ValueError(f'{args.epochs = } must be positive')

    if args.seed is None:
        args.seed = random.choice([7, 42, 69, 666, 420, 42069, 69420])

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
        train_generator_every=args.train_generator_every,
    )

    print(f'{params = }')

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.use_deterministic_algorithms(True)  # Needed for reproducible results
    logger.info(f'{torch.cuda.is_available() = }')
    device = select_cuda_device_by_memory() or torch.device('cpu')
    logger.info(f'{device = } {torch.cuda.device_count() = }')

    g = Generator(params.latent_sz, params.n_classes, params.leaky_relu_negative_slope).to(device)
    d = Discriminator(params.leaky_relu_negative_slope).to(device)

    # TODO: make calculation of learnable parameters more accurate
    def format_as_gibibytes(n: int) -> str:
        return f'{n / (1024 * 1024 * 1024):.2f} GiB'

    # logger.debug(f'{g = }')
    # logger.debug(f'{d = }')
    g.train()
    d.train()
    g_learnable_parameters = sum(p.numel() for p in g.parameters() if p.requires_grad)
    # g_learnable_parameters_in_bytes = g_learnable_parameters * 4
    d_learnable_parameters = sum(p.numel() for p in d.parameters() if p.requires_grad)
    # d_learnable_parameters_in_bytes = d_learnable_parameters * 4
    print(f'{g = }')
    # torchinfo.summary(g, (params.batch_size, params.latent_sz + params.n_classes, 1, 1))
    print(f'{format_as_gibibytes(g_learnable_parameters * 4) = }')
    print(f'{d = }')
    # torchinfo.summary(d, (params.batch_size, 2, 128, 512))
    print(f'{format_as_gibibytes(d_learnable_parameters * 4) = }')

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
    # test_dataset = DrumsDataset(test_dir)
    # val_dataset = DrumsDataset(val_dir)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=2
    )
    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=params.batch_size, shuffle=True, num_workers=2
    # )
    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=params.batch_size, shuffle=True, num_workers=2
    # )

    print_cuda_memory_usage()

    run: neptune.Run = neptune.init_run(
        name=args.name,
        project='jenner/deep-learning-final-project',
        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNzNkMDgxNS1lOTliLTRjNWQtOGE5Mi1lMDI5NzRkMWFjN2MifQ==',
        source_files=[__file__, './model.py'],
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
        params=params,
        run=run,
    )

    # logger.info('starting testing')
    # test(
    #     g=g,
    #     d=d,
    #     device=device,
    #     test_dataloader=test_dataloader,
    #     params=params,
    #     run=run,
    # )

    run.stop()

    return 0


if __name__ == '__main__':
    logger.debug(f'{sys.executable = }')
    main()

# %%
