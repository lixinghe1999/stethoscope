import scipy
import numpy as np
import matplotlib.pyplot as plt
def plot_spectrogram(title, w, fs, ax):
    ff, tt, Sxx = scipy.signal.spectrogram(w, fs=fs, nperseg=256, nfft=512)
    ax.pcolormesh(tt, ff, Sxx, cmap='gray_r', shading='gouraud')
    ax.set_title(title)
    ax.set_xlabel('t (sec)')
    ax.set_ylabel('Frequency (Hz)')
    ax.grid(True)
fs = 44100
T = 10
t = np.arange(0, int(T*fs)) / fs
chirp = scipy.signal.chirp(t, f0=25, f1=800, t1=T, method='linear')

fig, axs = plt.subplots(2, 1)
plot_spectrogram('Linear Chirp', chirp, fs, axs[0])
scipy.io.wavfile.write('chirp_playback.wav', fs, chirp)

fs, heartbeat = scipy.io.wavfile.read('heartbeat.wav')
b, a = scipy.signal.butter(4, [25, 800], 'bandpass', fs=fs)
heartbeat = scipy.signal.filtfilt(b, a, heartbeat)
heartbeat = (heartbeat - np.mean(heartbeat))/np.max(np.abs(heartbeat))
plot_spectrogram('Heartbeat', heartbeat, fs, axs[1])
scipy.io.wavfile.write('heartbeat_playback.wav', fs, heartbeat)
plt.show()