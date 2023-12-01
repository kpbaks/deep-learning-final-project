import numpy as np
import scipy.signal as dsp
from fs import AudioFile, FileStream
from typing import Iterator
from util import unwrap
import scipy.signal as dsp


def detrend(files: FileStream) -> FileStream:
    for file in files:
        file.data = dsp.detrend(file.data)
        yield file


def normalize(files: FileStream) -> FileStream:
    for file in files:
        max_val = np.max(np.abs(file.data))
        file.data = file.data / max_val
        yield file
