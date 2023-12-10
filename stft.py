#!/usr/bin/env -S pixi run python3
import numpy as np
import scipy
import torch


# a relatively large epsilon makes things more precise
EPSILON = 1.0e-3


def audio_2_spectrum(
    audio: np.ndarray,
    sample_rate: float,
    nperseg=256,
    use_mel_spectrum=False,
    use_instantaneous_frequency=True,
    unwrap_phase=True,
) -> torch.Tensor:
    """Convert a single audio waveform to the spectrum representation used by
    the network.
    """
    assert audio.shape[0] == 128 * 512

    # casting to a higher resolution to maximize quality
    audio = audio.astype(np.float64)

    frequencies, times, Zxx = scipy.signal.stft(
        audio, fs=sample_rate, nfft=nperseg, nperseg=nperseg, padded=False
    )
    assert Zxx.shape == (129, 513)

    # Slice Zxx to have shape (128, 512)
    # this gets rid of the highest frequency component
    # Cut off from 513 to 256
    Zxx = Zxx[:128, :256]
    assert Zxx.shape == (128, 256)

    magnitude_spectrum = np.abs(Zxx)
    phase_spectrum = np.angle(Zxx)
    assert magnitude_spectrum.shape == phase_spectrum.shape

    # TODO: mel spectrum here

    log_magnitude_spectrum = np.log(magnitude_spectrum + EPSILON)

    if use_instantaneous_frequency:
        phase_spectrum = _instantaneous_frequency(phase_spectrum, unwrap_phase)
    scaled_phase_spectrum = phase_spectrum / np.pi

    # magic numbers make it so that the log sits roughly between [-1, 1]
    scaled_log_magnitude_spectrum = -1.0 + (log_magnitude_spectrum + 6.90775) * 0.28

    # convert to tensor
    spectrogram = np.stack((scaled_log_magnitude_spectrum, scaled_phase_spectrum), axis=0)
    spectrogram = torch.from_numpy(spectrogram).to(torch.float32)
    assert spectrogram.shape == (2, *magnitude_spectrum.shape)

    return spectrogram


def spectrum_2_audio(
    spectrogram: torch.Tensor,
    sample_rate: float,
    nperseg=256,
    use_mel_spectrum=False,
    use_instantaneous_frequency=True,
    unwrap_phase=True,
):
    scaled_log_magnitude_spectrum = spectrogram[0]
    scaled_phase_spectrum = spectrogram[1]

    # unnormalize
    log_magnitude_spectrum = ((scaled_log_magnitude_spectrum + 1.0) / 0.28) - 6.90775
    phase_spectrum = scaled_phase_spectrum * np.pi

    # unlog
    magnitude_spectrum = np.exp(log_magnitude_spectrum) - EPSILON

    # uninstantaneous freq
    if use_instantaneous_frequency:
        phase_spectrum = np.cumsum(phase_spectrum, axis=1)

    Zxx = magnitude_spectrum * torch.exp(1j * phase_spectrum)
    assert Zxx.shape == (128, 256)
    # pad with the stuff we removed
    Zxx_padded = np.pad(Zxx, ([0, 1], [0, 513 - 256]))
    assert Zxx_padded.shape == (129, 513)

    _, audio = scipy.signal.istft(Zxx_padded, fs=sample_rate, nperseg=nperseg)

    return audio


def _unwrap_phases(phases: np.ndarray) -> np.ndarray:
    """Unwrap cyclic phase, so that it doesn't have discontinuities"""

    # diffs the wrapped phase, then transforms the diffs and takes
    # the integral again, so that it is unwrapped. It's kinda magical.
    dd = _diff_phases(phases)
    ddmod = np.fmod(dd + np.pi, 2.0 * np.pi) - np.pi
    idx = np.logical_and(ddmod == -np.pi, dd > 0)
    ddmod = np.where(idx, np.ones_like(ddmod) * np.pi, ddmod)
    ph_correct = ddmod - dd
    idx = np.abs(dd) < np.pi
    ddmod = np.where(idx, np.zeros_like(ddmod), dd)
    ph_cumsum = np.cumsum(ph_correct, axis=1)

    # prepend a zero-column
    # ph_cumsum = np.concatenate([np.zeros((phases.shape[0], 1)), ph_cumsum], axis=1)
    # print("ph_cumsum", ph_cumsum.shape)
    assert ph_cumsum.shape == phases.shape
    unwrapped = phases + ph_cumsum
    assert unwrapped.shape == phases.shape
    return unwrapped


def _diff_phases(phases: np.ndarray) -> np.ndarray:
    d = np.diff(phases, axis=1, prepend=0.0)
    assert d.shape == phases.shape
    return d


# Port of the original magenta GANsynth from tensorflow to pytorch
def _instantaneous_frequency(phases, use_unwrap=True):
    input_shape = phases.shape

    if use_unwrap:
        # allow phases to exceed bounds [-pi, pi]
        phase_unwrapped = _unwrap_phases(phases)
        dphase = _diff_phases(phase_unwrapped)
    else:
        dphase = _diff_phases(phases)
        # ensure that phases are bound between [-pi, pi] after
        # taking the finite difference
        dphase = np.where(dphase > np.pi, dphase - 2 * np.pi, dphase)
        dphase = np.where(dphase < -np.pi, dphase + 2 * np.pi, dphase)

    assert dphase.shape == input_shape
    return dphase


if __name__ == '__main__':
    from scipy.io import wavfile
    from pathlib import Path
    import matplotlib.pyplot as plt
    import random

    dataset_dir = Path.home() / 'datasets' / 'drums' / 'train'
    random_sample = random.choice(list(dataset_dir.glob('**/*.wav')))

    # with Path('./_temp/clean/y2k-core_clean/0_chat_stylized.wav').open('rb') as fh:
    with random_sample.open('rb') as fh:
        fs, audio = wavfile.read(fh)

    spec = audio_2_spectrum(audio, fs, use_instantaneous_frequency=True, unwrap_phase=True)
    plt.imshow(spec[1], origin='lower')
    plt.show()
    audio_out = spectrum_2_audio(spec, fs, use_instantaneous_frequency=True)

    with Path('./pivelyd_lmao.wav').open('wb') as fh:
        wavfile.write(fh, fs, audio_out)
