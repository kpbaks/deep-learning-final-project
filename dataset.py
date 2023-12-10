#!/usr/bin/env -S pixi run python3 -O
# %%

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import sys

import scipy
import torch
from sklearn.preprocessing import MinMaxScaler

from stft import audio_2_spectrum


@dataclass
class Metadata:
    id: int
    drum_type: str


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
        if not dataset_dir.exists():
            raise ValueError(f'{dataset_dir} does not exist')
        if not dataset_dir.is_dir():
            raise ValueError(f'{dataset_dir} is not a directory')

        self.wav_files = list(dataset_dir.glob('*.wav'))
        if len(self.wav_files) == 0:
            raise ValueError(f'{dataset_dir = } does not contain any WAV files')

        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    @staticmethod
    def parse_filename(filename: Path) -> Metadata:
        words = filename.stem.split('_')
        assert len(words) >= 2, f'filename {filename} does not match the pattern'
        id = int(words[0])
        drum_type = words[1]
        return Metadata(id=id, drum_type=drum_type)

    @staticmethod
    def onehot_encode_label(label: str) -> torch.Tensor:
        """
        One-hot encode the label.
        """
        match label:
            case 'kick':
                return torch.tensor([[1, 0, 0, 0]])
            case 'snare':
                return torch.tensor([[0, 1, 0, 0]])
            case 'chat':
                return torch.tensor([[0, 0, 1, 0]])
            case 'ohat':
                return torch.tensor([[0, 0, 0, 1]])
            case _:
                print(f'unknown label {label = }')
                raise ValueError(f'unknown label {label}')

    def __len__(self) -> int:
        return len(self.wav_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        # Load audio file
        if not 0 <= idx < len(self):
            raise IndexError(f'index {idx} is out of range')

        sample_rate, data = scipy.io.wavfile.read(self.wav_files[idx])
        # print(f'{sample_rate = } {data.shape = } {len(data) / sample_rate = } seconds')
        # sys.exit(0)

        spectrogram = audio_2_spectrum(data, sample_rate)

        metadata = self.parse_filename(self.wav_files[idx])
        return spectrogram, metadata.drum_type


def main(dataset_dir: Path) -> int:
    # import argparse
    # import os
    import random

    import matplotlib.pyplot as plt

    # import sys
    from loguru import logger

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
    spectrogram = dataset[idx][0]
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
    dataset_dir = Path.home() / 'datasets' / 'drums' / 'train'
    # print([i for i in range(2**10) if is_power_of_2(i)])
    sys.exit(main(dataset_dir))

# %%
