import scipy
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
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

files = os.listdir('dataset')
fname = 'chirp'
playback, sr = librosa.load('dataset/' + fname + '.wav', sr=None)
files_filter = [f for f in files if f.split('_')[0] == fname]
fig, ax = plt.subplots(1+(files_filter), 1, figsize=(3, 6))
plt.tight_layout()
freq_playback, response_playback, snr_playback = plot_spectrogram('Playback', playback, sr, ax[0], 0)

for i, f in enumerate(files_filter):
    _, phone, textile = f.split('_')
    record, sr = librosa.load('dataset/' + f, sr=None)[0]
    freq, response, snr = plot_spectrogram('_'.join([phone, textile]), playback, sr, ax[i+1], 0)
# plt.savefig('chirp.png', dpi=300)
plt.show()
