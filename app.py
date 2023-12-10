#!/usr/bin/env -S pixi run python3
# %%
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import neptune
import torch
from loguru import logger
from scipy.io import wavfile

from dataset import DrumsDataset
from model import Generator
from stft import spectrum_2_audio

parser = argparse.ArgumentParser(prog=os.path.basename(__file__))
parser.add_argument('--run-id', type=str, required=True)
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument(
    '--drum-type', type=str, choices=['kick', 'snare', 'chat', 'ohat'], default='snare'
)

args = parser.parse_args()
print(f'{args = }')

logger.info(f'connecting to neptune run {args.run_id}')
run: neptune.Run = neptune.Run(
    with_id=args.run_id,
    project='jenner/deep-learning-final-project',
    api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNzNkMDgxNS1lOTliLTRjNWQtOGE5Mi1lMDI5NzRkMWFjN2MifQ==',
    # mode='read-only'
)

# epoch = 1930
model_weights = Path.cwd() / 'models' / args.run_id / f'epoch_{args.epoch}' / 'generator.pth'
assert model_weights.exists(), f'{model_weights} does not exist'
# path = './models/DEEP-182/epoch_100/generator.pth'
model = Generator(256, 4, 0.2)
logger.info(f'loading model weights from {model_weights}')
model.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
model.eval()

latent = torch.randn(260)
drum_type = 'kick'
drum_type_one_hot = DrumsDataset.onehot_encode_label(drum_type)
latent[256:] = drum_type_one_hot
# latent[-4] = 1.0
# latent[:256] = torch.rand(256)

logger.info(f'generating audio for {args.drum_type}')
audio_stft = model(latent.reshape(1, 260, 1, 1)).detach().squeeze()
audio_stft_amp = audio_stft[0]

fig, axis = plt.subplots(2, 1, figsize=(8, 6))
axis[0].set_title(f'run_id: {args.run_id} epoch: {args.epoch} {args.drum_type} generated audio')
im = axis[0].imshow(audio_stft_amp, origin='lower')
fig.colorbar(im, ax=axis[0])

audio = spectrum_2_audio(audio_stft, 44100.0)
axis[1].plot(audio)
axis[1].set_title('audio')

output_file_path = Path.cwd() / 'generated' / f'{args.run_id}_{args.epoch}_{drum_type}'
output_file_path.parent.mkdir(parents=True, exist_ok=True)

# plt.imshow(audio_stft_amp)
# plt.show()
# audio = spectrum_2_audio(audio_stft, 44100.0)
# plt.plot(audio)
# plt.show()
with Path(f'{output_file_path}.wav').open('wb') as fh:
    wavfile.write(fh, 44100, audio)

with Path(f'{output_file_path}.png').open('wb') as fh:
    fig.savefig(fh, dpi=120)

logger.info('uploading generated audio to neptune')
run[f'generated_audio/{output_file_path.stem}.png'].upload(str(output_file_path) + '.png')
run[f'generated_audio/{output_file_path.stem}.wav'].upload(str(output_file_path) + '.wav')

logger.info('done')
