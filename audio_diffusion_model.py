#!/usr/bin/env -S pixi run python3 -O

# %%

import os
import sys
import time
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import optimize
import torch
import torchaudio
from loguru import logger
from tqdm import tqdm, trange

try:
    from rich import pretty, print
    pretty.install()
except ImportError or ModuleNotFoundError:
    pass

from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

SAMPLE_RATE: int = 16000
BATCH_SIZE: int = 12
NUM_SAMPLES: int = 2**18

# %%
class Timer:
    def __init__(self, message: str):
        self.message = message
        
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.message is not None:
            print(f"timer \"{self.message}\" took {self.interval:.4f} seconds")
        else:
            print(f"timer took {self.interval:.4f} seconds")


class DrumsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.wav_files = list(dataset_dir.glob("*.wav"))

    def __len__(self) -> int:
        return len(self.wav_files)

    def __getitem__(self, idx: int):
        # Load audio file
        sample_rate, audio = scipy.io.wavfile.read(self.wav_files[idx]) # [length, in_channels]
        assert isinstance(sample_rate, int), f"{type(sample_rate) = }"
        assert isinstance(audio, np.ndarray), f"{type(audio) = }"
        assert audio.ndim == 1, f"{audio.ndim = }" # must be mono
        # print(f"{sample_rate = }")
        # audio = audio.astype(np.float32) / 32767.0
        audio = audio.T # [in_channels, length]
        audio = audio.astype(np.float32) / 32767.0 # Why divide by 32767.0?
        audio = torch.from_numpy(audio)

        print(f"{audio.shape = }")

        return audio

# dataset = DrumsDataset(DATASET_DIR)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
# %%

# with Timer("Allocate model"):
#     model = DiffusionModel(
#         net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
#         in_channels=2, # U-Net: number of input/output (audio) channels
#         channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
#         factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
#         items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
#         attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
#         attention_heads=8, # U-Net: number of attention heads per attention item
#         attention_features=64, # U-Net: number of attention features per attention item
#         diffusion_t=VDiffusion, # The diffusion method used
#         sampler_t=VSampler, # The diffusion sampler used
#     ).to(device)

    # print(f"{model = }")


# %% 
# Train

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# with Timer("Train model"):
#     model.train()
#     num_epochs: int = 10
#     for epoch in trange(num_epochs):
#         for i, audio in enumerate(dataloader):
#             print(f"{i = }")
#             print(f"{audio.shape = }")
#             # audio = audio.to(device)
#             optimizer.zero_grad()

#             loss = model(audio.to(device))
#             loss.backward()

#             optimizer.step()


# %%

# num_channels: int = 1
# length: int = 2**14 # 16384

# with Timer("Test diffusion model"):
#     num_samples_to_generate: int = 5
#     for i in range(num_samples_to_generate):
#         # Turn noise into new audio sample with diffusion
#         noise = torch.randn(1, num_channels, length) # [batch_size, in_channels, length]
#         sample = model.sample(noise.to(device), num_steps=10) # Suggested num_steps 10-100
#         sample = sample.squeeze().cpu().numpy() # [length, in_channels]
#         sample = sample.T # [in_channels, length] scipy.io.wavfile.write expects this
#         scipy.io.wavfile.write(f"sample_{i}.wav", 16000, sample)

# %%

# with Timer("Test diffusion model"):

    
#     # length: int = 2**18 # 262144
#     length: int = 2**16 # 65536

#     # Train model with audio waveforms
#     audio = torch.randn(1, 2, length) # [batch_size, in_channels, length]
#     loss = model(audio.to(device))
#     loss.backward()

#     # Turn noise into new audio sample with diffusion
#     noise = torch.randn(1, 2, length) # [batch_size, in_channels, length]
#     sample = model.sample(noise.to(device), num_steps=10) # Suggested num_steps 10-100

#     # Save sample as WAV file
#     sample = sample.squeeze().cpu().numpy() # [length, in_channels]
#     sample = sample.T # [in_channels, length] scipy.io.wavfile.write expects this
#     print(f"{sample.shape = }")
#     scipy.io.wavfile.write("sample.wav", 16000, sample.astype(np.int16))


def create_model() -> DiffusionModel:
    model = DiffusionModel(
        net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
        in_channels=2, # U-Net: number of input/output (audio) channels
        channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
        factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
        items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
        attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
        attention_heads=8, # U-Net: number of attention heads per attention item
        attention_features=64, # U-Net: number of attention features per attention item
        diffusion_t=VDiffusion, # The diffusion method used
        sampler_t=VSampler, # The diffusion sampler used
    )
    return model    

def parse_argv(argv: list[str]) -> argparse.Namespace:
    prog: str = os.path.basename(__file__).removesuffix(".py")
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--dataset-dir", type=Path, default=Path.home() / "datasets" / "drums")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args(argv[1:])
    assert args.dataset_dir.exists(), f"{args.dataset_dir} does not exist"
    assert args.dataset_dir.is_dir(), f"{args.dataset_dir} is not a directory"

    return args

def main(argc: int, argv: list[str]) -> int:
    args = parse_argv(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"{device = }")
    logger.info(f"{device = }")

    DATASET_DIR = Path.home() / "datasets" / "drums"
    assert DATASET_DIR.exists(), f"{DATASET_DIR} does not exist"
    assert DATASET_DIR.is_dir(), f"{DATASET_DIR} is not a directory"
    logger.info(f"{DATASET_DIR = }")

    dataset = DrumsDataset(DATASET_DIR)
    logger.info(f"{len(dataset) = }")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    model = create_model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{num_params = }")

    model.train()
    num_epochs: int = 10
    for epoch in trange(num_epochs):
        for i, audio in enumerate(dataloader):
            print(f"{i = }")
            print(f"{audio.shape = }")
            # audio = audio.to(device)
            optimizer.zero_grad()

            loss = model(audio.to(device))
            loss.backward()

            optimizer.step()
    
    return 0

if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))
