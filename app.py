import torch
from model import Generator
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pathlib import Path
from stft import spectrum_2_audio

path = './models/DEEP-182/epoch_100/generator.pth'
model = Generator(256, 4, 0.2)
model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
model.eval()
latent = torch.empty(260)
latent[-4] = 1.0
latent[:256] = torch.rand(256)
audio_stft = model(latent.reshape(1, 260, 1, 1)).detach().squeeze()
audio_stft_amp = audio_stft[0]
plt.imshow(audio_stft_amp)
plt.show()
audio = spectrum_2_audio(audio_stft, 44100.0)
plt.plot(audio)
plt.show()
with Path('pivelyd.wav').open('wb') as fh:
    wavfile.write(fh, 44100, audio)
