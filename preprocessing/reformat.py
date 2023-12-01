import numpy as np
import scipy.signal as dsp
from fs import AudioFile, FileStream
from typing import Iterator
from util import unwrap
import scipy.signal as dsp


def to_f32(files: FileStream) -> FileStream:
    for file in files:
        file.data = file.data.astype(np.float32)
        yield file


def to_mono(files: FileStream) -> FileStream:
    for file in files:
        file.data = np.sum(file.data, axis=-1)
        yield file


def resample(files: FileStream, sr=44100) -> FileStream:
    for file in files:
        resampling_factor = sr / unwrap(file.sample_rate)
        new_length = int(file.data.shape[0]) * resampling_factor
        resampled_data = dsp.resample(file.data, new_length)
        if resampled_data is not np.ndarray:
            raise TypeError("infallible")
        file.data = resampled_data
        yield file


def trim_pad(files: FileStream, size: int) -> FileStream:
    for file in files:
        # trim
        file.data = file.data[:size]

        # pad
        if (cur_width := file.data.shape[0]) < size:
            pad_width = size - cur_width
            file.data = np.pad(file.data, (0, pad_width))

        # post condition 🥴
        assert file.data.shape[0] == size
        yield file


def fade_end(files: FileStream, fade_n: int) -> FileStream:
    for file in files:
        fade_values = np.linspace(1.0, 0.0, fade_n)
        file.data[-fade_n:] *= fade_values
        yield file
