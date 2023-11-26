#!/usr/bin/env -S pixi run python3 -O

from pathlib import Path

import numpy as np
import scipy
import torch
from sklearn.preprocessing import MinMaxScaler


class DrumsDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for our custom drums dataset.
    """

    def __init__(self, dataset_dir: Path):
        """
        :param dataset_dir: Path to the dataset directory
        """
        self.dataset_dir = dataset_dir
        assert dataset_dir.exists(), f"{dataset_dir} does not exist"
        assert dataset_dir.is_dir(), f"{dataset_dir} is not a directory"

        self.wav_files = list(dataset_dir.glob("*.wav"))
        self.scaler = MinMaxScaler()

    def __len__(self) -> int:
        return len(self.wav_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Load audio file
        if not 0 <= idx < len(self):
            raise IndexError(f"index {idx} is out of range")
        sample_rate, data = scipy.io.wavfile.read(self.wav_files[idx])

        # Apply Short-Time Fourier Transform (STFT)
        _, _, Zxx = scipy.signal.stft(data, fs=sample_rate)

        # Get the magnitude of the spectrogram
        spectrogram = np.abs(Zxx)

        # Normalize the spectrogram
        spectrogram = self.scaler.fit_transform(spectrogram)

        # Add a channel dimension (1, height, width)
        spectrogram = spectrogram[np.newaxis, ...]

        # Convert to torch tensor
        spectrogram = torch.from_numpy(spectrogram)

        return spectrogram


def main() -> int:
    import argparse
    import os
    import random
    import sys

    from loguru import logger
    import matplotlib.pyplot as plt

    def parse_argv(argv: list[str]) -> argparse.Namespace:
        prog: str = os.path.basename(__file__)
        argv_parser = argparse.ArgumentParser(prog=prog)
        argv_parser.add_argument(
            "dataset_dir", type=str, help="path to dataset directory"
        )
        args = argv_parser.parse_args(argv)
        return args

    args = parse_argv(sys.argv[1:])

    dataset_dir = Path(args.dataset_dir)
    logger.debug(f"{dataset_dir = }")

    assert dataset_dir.exists(), f"{dataset_dir} does not exist"
    assert dataset_dir.is_dir(), f"{dataset_dir} is not a directory"
    dataset = DrumsDataset(dataset_dir)
    logger.info(f"{len(dataset) = }")

    # Get a random sample
    idx = random.randint(0, len(dataset))
    spectrogram = dataset[idx]
    logger.info(f"{spectrogram.shape = }")

    # Plot the spectrogram
    logger.info(f"plotting spectrogram of sample {idx = }")

    plt.figure()
    plt.imshow(spectrogram[0])
    plt.title(f"{dataset_dir.name} - {idx}")
    plt.savefig(f"{dataset_dir.name}-{idx}.png", dpi=80)
    plt.show()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
