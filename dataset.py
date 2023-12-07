#!/usr/bin/env -S pixi run python3 -O
# %%

from pathlib import Path

import numpy as np
import scipy
import torch
from sklearn.preprocessing import MinMaxScaler


def is_power_of_2(n: int) -> bool:
    """
    Check if a number is a power of 2.
    """
    return (n & (n - 1) == 0) and n != 0


class DrumsDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for our custom drums dataset.
    """

    def __init__(self, dataset_dir: Path):
        """
        :param dataset_dir: Path to the dataset directory
        """
        self.dataset_dir = dataset_dir
        assert dataset_dir.exists(), f'{dataset_dir} does not exist'
        assert dataset_dir.is_dir(), f'{dataset_dir} is not a directory'

        self.wav_files = list(dataset_dir.glob('*.wav'))
        self.scaler = MinMaxScaler()

    def __len__(self) -> int:
        return len(self.wav_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Load audio file
        if not 0 <= idx < len(self):
            raise IndexError(f'index {idx} is out of range')
        sample_rate, data = scipy.io.wavfile.read(self.wav_files[idx])

        # print(f'{data.shape =}')
        # Apply Short-Time Fourier Transform (STFT)
        frequencies, times, Zxx = scipy.signal.stft(
            data, fs=sample_rate, nfft=256, nperseg=256, noverlap=128, padded=False
        )

        # Slice Zxx to have shape (128, 512)
        Zxx = Zxx[:128, :512]

        # print(f'{Zxx.shape = }')
        # Get the magnitude of the spectrogram
        magnitude_spectrum = np.abs(Zxx)
        phase_spectrum = np.angle(Zxx)
        assert magnitude_spectrum.shape == phase_spectrum.shape

        # Normalize the spectrogram
        # spectrogram = self.scaler.fit_transform(spectrogram)

        # Stack the magnitude and phase spectrograms
        # Put the channels first, because PyTorch expects it that way
        spectrogram = np.stack((magnitude_spectrum, phase_spectrum), axis=0)
        assert spectrogram.shape == (2, *magnitude_spectrum.shape)

        # Convert to torch tensor
        spectrogram = torch.from_numpy(spectrogram)

        return spectrogram


def main(dataset_dir: Path) -> int:
    # import argparse
    # import os
    import random
    # import sys

    from loguru import logger
    import matplotlib.pyplot as plt

    # def parse_argv(argv: list[str]) -> argparse.Namespace:
    #     prog: str = os.path.basename(__file__)
    #     argv_parser = argparse.ArgumentParser(prog=prog)
    #     argv_parser.add_argument(
    #         "dataset_dir", type=str, help="path to dataset directory"
    #     )
    #     args = argv_parser.parse_args(argv)
    #     return args

    # args = parse_argv(sys.argv[1:])

    # dataset_dir = Path(args.dataset_dir)
    logger.debug(f'{dataset_dir = }')

    assert dataset_dir.exists(), f'{dataset_dir} does not exist'
    assert dataset_dir.is_dir(), f'{dataset_dir} is not a directory'
    dataset = DrumsDataset(dataset_dir)
    # logger.info(f'{len(dataset) = }')

    # Get a random sample
    idx = random.randint(0, len(dataset))
    spectrogram = dataset[idx]
    # logger.info(f'{spectrogram.shape = }')

    # Plot the spectrogram
    logger.info(f'plotting spectrogram of sample {idx = }')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].pcolormesh(spectrogram[0])
    axes[1].pcolormesh(spectrogram[1])
    # plt.figure()
    # plt.pcolormesh(spectrogram[1])
    # plt.imshow(spectrogram[0])
    plt.title(f'{dataset_dir.name} - {idx}')
    # plt.savefig(f"{dataset_dir.name}-{idx}.png", dpi=80)
    plt.show()

    return 0


if __name__ == '__main__':
    import sys

    dataset_dir = Path.home() / 'datasets' / 'classic_clean'
    # print([i for i in range(2**10) if is_power_of_2(i)])
    sys.exit(main(dataset_dir))
