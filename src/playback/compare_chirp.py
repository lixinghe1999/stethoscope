import scipy
import numpy as np
import matplotlib.pyplot as plt
import librosa
import warnings
warnings.filterwarnings("ignore")

def plot_spectrogram(title, w, fs, ax, offset=0):
    ff, tt, Sxx = scipy.signal.spectrogram(w, fs=fs, nperseg=2048, nfft=4096)
    Sxx = np.log10(Sxx) * 10
    chirp_freq = (tt-offset) * (800-25)/10
    chirp_freq = np.clip(chirp_freq, 25, 800)
    chirp_index = np.round(chirp_freq / (fs/2) * len(ff)).astype(int)
    
    frequency_response = Sxx[chirp_index, np.arange(len(chirp_index))]
    # print(frequency_response)
    ax.pcolormesh(tt, ff, Sxx)
    ax.plot(tt, chirp_freq, c='r')
    ax.set_title(title)
    ax.set_ylim([0, 2000])
    ax.set_xlim([offset, offset+10])
    return chirp_freq, frequency_response

smartphone, sr = librosa.load('data/chirp_smartphone_contact.wav', sr=None)
stethoscope1, sr = librosa.load('data/chirp_stethoscope.wav', sr=None)

fig, ax = plt.subplots(4, 1, figsize=(3, 6))
plt.tight_layout()
freq1, FR1 = plot_spectrogram('smartphone', smartphone, sr, ax[0], 0.4)
freq2, FR2 = plot_spectrogram('stethoscope', stethoscope1, sr, ax[1], 0 )
# freq3, FR3 = plot_spectrogram('stethoscope', stethoscope2, sr, ax[2, 0], 0.5)

ax[2].plot(freq1, FR1, label='smartphone')
ax[2].set_ylim([-100, -10])
ax[2].set_title('frequency response')
ax[2].plot(freq2, FR2, label='stethoscope')
ax[2].legend()

align_freq = np.linspace(0, 800, 100)
FR1_aligned = np.interp(align_freq, freq1, FR1)
FR2_aligned = np.interp(align_freq, freq2, FR2)
ax[3].plot(align_freq, FR2_aligned - FR1_aligned)
ax[3].set_title('difference')
ax[3].set_ylim([0, 40])

plt.savefig('compare_chirp.png', dpi=300)