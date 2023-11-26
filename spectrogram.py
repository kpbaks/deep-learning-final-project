# %%
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile


def plot_spectrogram(file_path, max_freq=None):
    sampling_rate, data = wavfile.read(file_path)

    # Apply Short-Time Fourier Transform (STFT)
    f, t, Zxx = stft(data, fs=sampling_rate, nperseg=256, noverlap=128)

    # Limit the frequency range
    if max_freq is not None:
        freq_index = f <= max_freq
        f = f[freq_index]
        Zxx = Zxx[freq_index, :]

    # Plot the spectrogram
    plt.figure(figsize=(14, 5))
    plt.pcolormesh(t, f, np.abs(Zxx), shading="gouraud")
    plt.title("Spectrogram (Limited Frequency)")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.colorbar(format="%+2.0f dB")
    plt.ylim(0, max_freq if max_freq is not None else None)
    plt.show()


# Call the function with the path to your audio file and the maximum frequency of interest

DATASET_DIR = Path.home() / "datasets" / "drums"
wav_files = list(DATASET_DIR.glob("*.wav"))

plot_spectrogram(str(wav_files[0]), max_freq=5000)  # Adjust max_freq as needed
