# %%
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# import torch

# import torchaudio
import librosa

# torchaudio
# %%
DATASET_DIR = Path.home() / 'datasets' / 'drums'
wav_files = list(DATASET_DIR.glob('*.wav'))


# Load audio file
data, sample_rate = librosa.load(wav_files[0], sr=None)  # sr=None means no resampling

print(f'sample_rate: {sample_rate}')
print(f'data.shape: {data.shape}')
# %%
# Plot the mel spectrogram

n_fft: int = 1024 * 4
win_length: int | None = None
hop_length: int = 512
n_mels: int = 64

mel_spectrogram = librosa.feature.melspectrogram(
    y=data,
    sr=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    n_mels=n_mels,
    htk=True,
    norm=None,
)

assert mel_spectrogram.shape[0] == n_mels

print(f'mel_spectrogram.shape: {mel_spectrogram.shape}')

# %%

plt.figure(figsize=(10, 4))
librosa.display.specshow(
    librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time'
)

plt.colorbar(format='%+2.0f dB')

plt.title('Mel spectrogram')


# %%

# Paragraph is taken from the GANSynth paper page 4:

# For spectral representations, we compute STFT magnitudes and phase angles using TensorFlow’s built-in implementation. We use an STFT with 256 stride and 1024 frame size, resulting in 75% frame overlap and 513 frequency bins. We trim the Nyquist frequency and pad in time to get an “image” of size (256, 512, 2). The two channel dimension correspond to magnitude and phase. We take the log of the magnitude to better constrain the range and then scale the magnitudes to be between -1 and 1 to match the tanh output nonlinearity of the generator network. The phase angle is also scaled to between -1 and 1 and we refer to these variants as “phase” models. We optionally unwrap the phase angle and take the finite difference as in Figure 1; we call the resulting models “instantaneous frequency” (“IF”) models. We also find performance is sensitive to having sufficient frequency resolution at the lower frequency range. Maintaining 75% overlap we are able to double the STFT frame size and stride, resulting in spectral images with size (128, 1024, 2), which we refer to as high frequency resolution, “+ H”, variants. Lastly, to provide even more separation of lower frequencies we transform both the log magnitudes and instantaneous frequencies to a mel frequency scale without dimensional compression (1024 bins), which we refer to as “IF-Mel” variants. To convert back to linear STFTs we just use the approximate inverse linear transformation, which, perhaps surprisingly does not harm audio quality significantly.

stride: int = 256
frame_size: int = 1024

stft = librosa.stft(
    data,
    n_fft=frame_size,
    hop_length=stride,
    win_length=win_length,
)


print(f'stft.shape: {stft.shape}')
assert stft.shape[0] == frame_size // 2 + 1 == 513, f'{stft.shape = }'


magnitude = np.abs(stft)
phase = np.angle(stft)

# Plot the magnitude spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(
    librosa.amplitude_to_db(magnitude, ref=np.max), y_axis='linear', fmax=8000, x_axis='time'
)

plt.colorbar(format='%+2.0f dB')

plt.title('Magnitude spectrogram')

# Plot the phase spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(phase, y_axis='linear', fmax=8000, x_axis='time')

plt.colorbar(format='%+2.0f dB')
