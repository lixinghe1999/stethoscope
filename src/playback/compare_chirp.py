import scipy
import numpy as np
import matplotlib.pyplot as plt
import librosa
import warnings
warnings.filterwarnings("ignore")

def plot_spectrogram(title, w, fs, ax, offset=0):
    b, a = scipy.signal.butter(4, [25, 8000], 'bandpass', fs=fs)
    w = scipy.signal.filtfilt(b, a, w)
    ff, tt, Sxx = scipy.signal.spectrogram(w, fs=fs, nperseg=2048, nfft=4096)
    chirp_freq = (tt-offset) * (800-25)/10
    chirp_freq = np.clip(chirp_freq, 25, 800)
    chirp_index = np.round(chirp_freq / (fs/2) * len(ff)).astype(int)
    # chirp_index = np.argmax(Sxx, axis=0)
    
    frequency_response = Sxx[chirp_index, np.arange(len(chirp_index))]
    energy = np.mean(Sxx, axis=0)
    snr = np.log10(frequency_response / energy) * 10
    frequency_response = np.log10(frequency_response) * 10
    Sxx = np.log10(Sxx) * 10
    ax.pcolormesh(tt, ff, Sxx)
    ax.plot(tt, ff[chirp_index], c='r')
    ax.set_title(title)
    ax.set_ylim([0, 1000])
    ax.set_xlim([offset, offset+10])
    return chirp_freq, frequency_response, snr

pixelxl, sr = librosa.load('data/chirp_pixelxl.wav', sr=None)
pixel6, sr = librosa.load('data/chirp_pixel6.wav', sr=None)
stethoscope1, sr = librosa.load('data/chirp_stethoscope.wav', sr=None)

fig, ax = plt.subplots(5, 1, figsize=(3, 6))
plt.tight_layout()
freq1, FR1, SNR1 = plot_spectrogram('Pixel XL', pixelxl, sr, ax[0], 0.4)
freq3, FR3, SNR3 = plot_spectrogram('Pixel 6', pixel6, sr, ax[1], 0 )

freq2, FR2, SNR2 = plot_spectrogram('stethoscope', stethoscope1, sr, ax[2], 0 )


ax[3].plot(freq1, FR1, label='Pixel XL')
ax[3].plot(freq3, FR3, label='Pixel 6')
ax[3].plot(freq2, FR2, label='stethoscope')
ax[3].set_ylim([-100, -20])
ax[3].set_title('frequency response')
ax[3].legend()


ax[4].plot(freq1, SNR1, label='Pixel XL')
ax[4].plot(freq3, SNR3, label='Pixel 6')
ax[4].plot(freq2, SNR2, label='stethoscope')
ax[4].set_ylim([0, 30])

ax[4].set_title('SNR')
ax[4].legend()


plt.savefig('chirp.png', dpi=300)
plt.show()
