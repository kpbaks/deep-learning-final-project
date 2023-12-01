from pathlib import Path
from typing import Iterator, Tuple
from scipy.io import wavfile
from dataclasses import dataclass
from itertools import chain
import numpy as np


STANDARD_CATEGORIES = ["chat", "clap", "cym", "kick", "ohat", "other", "snare", "tom"]
STANDARD_DEST_PATH = "_dataset"


@dataclass
class AudioFile:
    path: Path
    name: str
    data: np.ndarray | None = None
    sample_rate: int | None = None

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


def file_loader(base_path: str) -> Iterator[AudioFile]:
    for category in STANDARD_CATEGORIES:
        dir_ = Path(base_path) / category
        for file in dir_.glob("*.wav"):
            yield AudioFile(file, category).load_data()


def file_storer(base_path: str, file_stream: Iterator[AudioFile]):
    """Consumes the file_stream, writing it to a file"""
    for i, file in enumerate(file_stream):
        new_path = Path(base_path) / STANDARD_DEST_PATH / f"{file.name}_{i}.wav"
        file.store_data(new_path)


file_storer("./_temp", file_loader("./_temp"))
