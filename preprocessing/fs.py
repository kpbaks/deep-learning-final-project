from pathlib import Path
from typing import Iterator
from scipy.io import wavfile
import numpy as np
from util import unwrap
from copy import deepcopy


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
        self.data = np.copy(data)
        self.sample_rate = sample_rate
        self.labels = deepcopy(labels)

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
        tag = "_" + "_".join(file.labels)
        new_path = Path(base_path) / STANDARD_DEST_PATH / f"{i}_{file.name}{tag}.wav"
        file.store_data(new_path)


def label(file: AudioFile, label: str) -> AudioFile:
    file.labels.append(label)
    return file


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
