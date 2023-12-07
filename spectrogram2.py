# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft

import random
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

# import numpy as np
from scipy.signal import butter, sosfilt


DATASET_DIR = Path.home() / 'datasets' / 'classic_clean'
wav_files = list(DATASET_DIR.glob('*.wav'))

# Replace 'path_to_file.wav' with the path to your WAV file
# filename = 'path_to_file.wav'
filename = random.choice(wav_files)

# Load the WAV file
sample_rate, data = wavfile.read(filename)

# Compute the Short-Time Fourier Transform (STFT)
frequencies, times, Zxx = stft(data, fs=sample_rate)

minmaxscaler = MinMaxScaler()
Zxx = minmaxscaler.fit_transform(np.abs(Zxx))

# Plot the spectrogram
plt.pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
logger.info(
    f'{filename = } {data.shape = } {sample_rate = } {Zxx.shape = } {frequencies.shape = } {times.shape = }'
)

logger.info(f'{max(frequencies) = }')


# %%


DATASET_DIR = Path.home() / 'datasets' / 'classic_clean'
wav_files = list(DATASET_DIR.glob('*.wav'))

# Replace 'path_to_file.wav' with the path to your WAV file
# filename = 'path_to_file.wav'
filename = random.choice(wav_files)

# Load the WAV file
sample_rate, data = wavfile.read(filename)


# Function to apply a band-pass filter
def bandpass_filter(signal, lowcut, highcut, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    filtered_signal = sosfilt(sos, signal)
    return filtered_signal


# Assuming `data` is your audio data and `sample_rate` is the sampling rate

# Apply band-pass filter
filtered_data = bandpass_filter(data, 20, 20000, sample_rate)

# Compute STFT
frequencies, times, Zxx = stft(data, fs=sample_rate)

# Logarithmic scaling for amplitude
Zxx_log = np.log(np.abs(Zxx) + 1e-8)  # Adding a small constant to avoid log(0)

# Normalize the data
Zxx_normalized = (Zxx_log - np.min(Zxx_log)) / (np.max(Zxx_log) - np.min(Zxx_log))

# Plot the normalized logarithmic spectrogram
plt.pcolormesh(times, frequencies, Zxx_normalized, shading='gouraud')
plt.title('Normalized Log Spectrogram')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# %%
