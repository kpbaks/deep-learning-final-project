import numpy as np
import scipy.signal as dsp
from fs import AudioFile
from util import unwrap


def to_f32(file: AudioFile) -> AudioFile:
    file.data = file.data.astype(np.float32)
    return file


def to_mono(file: AudioFile) -> AudioFile:
    # if already mono do nothing
    if file.data.ndim == 1:
        return file
    file.data = np.sum(file.data, axis=-1)
    return file


def resample(file: AudioFile, sr=44100) -> AudioFile:
    resampling_factor = sr / unwrap(file.sample_rate)
    new_length = int(file.data.shape[0] * resampling_factor)
    resampled_data = dsp.resample(file.data, new_length)
    if not isinstance(resampled_data, np.ndarray):
        raise TypeError("infallible")
    file.data = resampled_data
    return file


def trim_pad(file: AudioFile, size: int) -> AudioFile:
    # trim
    file.data = file.data[:size]

    # pad
    if (cur_width := file.data.shape[0]) < size:
        pad_width = size - cur_width
        file.data = np.pad(file.data, (0, pad_width))

    # post condition 🥴
    assert file.data.shape[0] == size
    return file
