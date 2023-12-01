from pathlib import Path
from typing import Iterator, Tuple
from scipy.io import wavfile
from dataclasses import dataclass
from itertools import chain
import numpy as np
from util import unwrap


STANDARD_CATEGORIES = ["chat", "clap", "cym", "kick", "ohat", "other", "snare", "tom"]
STANDARD_DEST_PATH = "_dataset"


class AudioFile:
    def __init__(
        self,
        path: Path,
        name: str,
        data: np.ndarray = np.array([]),
        sample_rate: int | None = None,
        labels: list[str] = [],
    ):
        self.path = path
        self.name = name
        self.data = data
        self.sample_rate = sample_rate
        self.labels = labels

    def load_data(self):
        with self.path.open("rb") as fh:
            rate, audio = wavfile.read(fh)
            self.sample_rate = rate
            self.data = audio
        return self

    def store_data(self, path: Path):
        path.parents[0].mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            wavfile.write(fh, self.sample_rate, self.data)
        return self


FileStream = Iterator[AudioFile]


def file_loader(base_path: str) -> Iterator[AudioFile]:
    for category in STANDARD_CATEGORIES:
        dir_ = Path(base_path) / category
        for file in dir_.glob("*.wav"):
            yield AudioFile(unwrap(file), category).load_data()


def file_storer(base_path: str, file_stream: Iterator[AudioFile]):
    """Consumes the file_stream, writing it to a file"""
    for i, file in enumerate(file_stream):
        new_path = Path(base_path) / STANDARD_DEST_PATH / f"{file.name}_{i}.wav"
        file.store_data(new_path)


def head(file_stream: FileStream):
    for i, file in enumerate(file_stream):
        if i >= 100:
            break
        print("---")
        print(f"name: {file.name}")
        print(f"path: {file.path}")
        print(f"size: {file.data.shape}")
        print(f"sr: {file.sample_rate}")


head(file_loader("./_temp"))
