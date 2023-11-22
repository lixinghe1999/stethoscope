import os 
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
def spectrum(data1, data2):
    ps1 = np.abs(np.fft.fft(data1))
    ps2 = np.abs(np.fft.fft(data2))
    ps1 = np.fft.fftshift(ps1)[len(data1)//2:]
    ps2 = np.fft.fftshift(ps2)[len(data2)//2:]
    freq = np.fft.fftshift(np.fft.fftfreq(len(data1), 1/4000))[len(data1)//2:]
    return freq, ps1, ps2
directory = 'smartphone/PhysioNet/training-b' + '_processed'
files = os.listdir(directory)
files = [f for f in files if f.startswith('MIC')]
fig, ax = plt.subplots(1, 3)
response = []
freq_index = np.arange(0, 2000, 10)
for f in tqdm(files[:]):
    data, sr = librosa.load(os.path.join(directory, f), sr=4000, mono=False)
    record = data[0]
    playback = data[1]
    freq, ps1, ps2 = spectrum(record, playback)
    res = ps1 / ps2
    ax[0].plot(freq, ps1)
    ax[1].plot(freq, ps2)
    # ax[0].plot(freq, res)
    res = np.interp(freq_index, freq, res)
    response.append(res)
response = np.stack(response, axis=0)
response_mean = np.mean(response, axis=0)
response_std = np.std(response, axis=0)
plt.plot(freq_index, response_mean)
plt.fill_between(freq_index, response_mean - response_std, response_mean + response_std, alpha=0.3)
plt.show()


