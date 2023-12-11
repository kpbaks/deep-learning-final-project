from stft import audio_2_spectrum

import scipy
import matplotlib.pyplot as plt

fs, audio = scipy.io.wavfile.read('_temp/clean/y2k-core_clean/610_snare_stylized.wav')
audio_stft = audio_2_spectrum(audio, fs)
audio_stft_amp = audio_stft[0]

fig = plt.figure(figsize=(8, 6))
ax00 = plt.subplot2grid((2, 2), (0, 0), fig=fig)
ax00.set_title('Magnitude Spectrum')
ax00.imshow(audio_stft_amp, origin='lower')
ax01 = plt.subplot2grid((2, 2), (0, 1), fig=fig)
# Plot the magnitude spectrum in the upper left corner
ax01.set_title('Phase Spectrum')
ax01.imshow(audio_stft[1], origin='lower')
# Plot the time series in the lower left corner
ax10 = plt.subplot2grid((2, 2), (1, 0), colspan=2, fig=fig)
ax10.set_title('Time Series')
ax10.plot(audio[: len(audio) // 2])
plt.show()
