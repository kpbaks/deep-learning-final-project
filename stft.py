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
):
    """Convert a single audio waveform to the spectrum representation used by
    the network.
    """
    assert audio.shape[0] == 256 * 512

    frequencies, times, Zxx = scipy.signal.stft(
        audio, fs=sample_rate, nfft=nperseg, nperseg=nperseg, padded=False
    )
    assert Zxx.shape == (129, 513)

    # Slice Zxx to have shape (128, 512)
    # this gets rid of the highest frequency component
    Zxx = Zxx[:128, :512]
    assert Zxx.shape == (128, 512)

    magnitude_spectrum = np.abs(Zxx)
    phase_spectrum = np.angle(Zxx)
    assert magnitude_spectrum.shape == phase_spectrum.shape

    # TODO: mel spectrum here

    log_magnitude_spectrum = np.log(magnitude_spectrum + EPSILON)

    if use_instantaneous_frequency:
        # TODO: instantaneous frequency
        pass

    # magic numbers make it so that the log sits roughly between [-1, 1]
    scaled_log_magnitude_spectrum = -1.0 + (log_magnitude_spectrum + 6.90775) * 0.28
    scaled_phase_spectrum = phase_spectrum / np.pi

    # convert to tensor
    spectrogram = np.stack((scaled_log_magnitude_spectrum, scaled_phase_spectrum), axis=0)
    spectrogram = torch.from_numpy(spectrogram)
    assert spectrogram.shape == (2, *magnitude_spectrum.shape)

    return spectrogram


def spectrum_2_audio(spectrogram, sample_rate, nperseg=256, overlap=128):
    scaled_log_magnitude_spectrum = spectrogram[0]
    scaled_phase_spectrum = spectrogram[1]

    # unnormalize
    log_magnitude_spectrum = ((scaled_log_magnitude_spectrum + 1.0) / 0.28) - 6.90775
    phase_spectrum = scaled_phase_spectrum * np.pi

    # unlog
    magnitude_spectrum = np.exp(log_magnitude_spectrum) - EPSILON

    Zxx = magnitude_spectrum * torch.exp(1j * phase_spectrum)
    assert Zxx.shape == (128, 512)
    # pad with the stuff we removed
    Zxx_padded = np.pad(Zxx, ([0, 1], [0, 1]))
    assert Zxx_padded.shape == (129, 513)

    _, audio = scipy.signal.istft(Zxx_padded, fs=sample_rate, nperseg=nperseg)

    return audio


def _unwrap_phases(phases):
    pass


def _diff_phases(phases):
    pass


if __name__ == '__main__':
    from scipy.io import wavfile
    from pathlib import Path

    with Path('./_temp/clean/y2k-core_clean/192_kick_stylized.wav').open('rb') as fh:
        fs, audio = wavfile.read(fh)

    spec = audio_2_spectrum(audio, fs)
    audio_out = spectrum_2_audio(spec, fs)

    with Path('./pivelyd_lmao.wav').open('wb') as fh:
        wavfile.write(fh, fs, audio_out)
