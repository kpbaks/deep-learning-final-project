import numpy as np
import scipy.signal as dsp
from fs import AudioFile


def detrend(file: AudioFile) -> AudioFile:
    file.data = dsp.detrend(file.data)
    return file


def fade_end(file: AudioFile, fade_n: int) -> AudioFile:
    fade_values = np.linspace(1.0, 0.0, fade_n)
    file.data[-fade_n:] = file.data[-fade_n:] * fade_values
    return file


def normalize(file: AudioFile) -> AudioFile:
    max_val = np.max(np.abs(file.data))
    file.data = file.data / max_val
    return file
