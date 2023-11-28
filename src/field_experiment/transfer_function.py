import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy
import soundfile as sf
import sys
sys.path.append('../')
from utils import *

import warnings
warnings.filterwarnings("ignore")
filter_list = {'heartbeat pure': scipy.signal.butter(4, 400, 'lowpass', fs=sr_mic),
               'heartbeat filter': scipy.signal.butter(4, [10, 500], 'bandpass', fs=sr_mic),
               'imu': scipy.signal.butter(4, [1, 100], 'bandpass', fs=sr_imu)}
sr_mic = 4000
sr_imu = 400
sr_ppg = 25
import os 
import librosa
import numpy as np
import matplotlib.pyplot as plt
def transfer_function(record, reference, ax):
    def spectrum(data1, data2):
        ps1 = np.abs(np.fft.fft(data1))
        ps2 = np.abs(np.fft.fft(data2))
        ps1 = np.fft.fftshift(ps1)[len(data1)//2:]
        ps2 = np.fft.fftshift(ps2)[len(data2)//2:]
        freq = np.fft.fftshift(np.fft.fftfreq(len(data1), 1/4000))[len(data1)//2:]
        return freq, ps1, ps2
    freq, ps1, ps2 = spectrum(record, reference)
    spectrum_scale = np.mean(ps1) / np.mean(ps2)
    ps2 = ps2 * spectrum_scale
    res = ps1 / ps2
    ax[0].plot(freq, ps1, label='record')
    ax[0].plot(freq, ps2, label='reference')
    ax[1].plot(freq, res)
    ax[0].legend()
    return freq, res

def visual_imu(data_imu, axs):
    '''
    two axes to show IMU data in waveform and STFT
    '''
    f, t, imu_stft = scipy.signal.stft(data_imu, axis=0, fs=sr_imu, nperseg=128,)
    t_imu = np.arange(len(data_imu)) / sr_imu
    axs[0].plot(t_imu, data_imu)
    axs[1].pcolormesh(t, f, np.abs(imu_stft))

def visual_mic(data_mic, axs):
    '''
    two axes to show microphone data in waveform and STFT, apply to smartphone or stethoscope recordings
    '''
    t_mic = np.arange(len(data_mic)) / sr_mic
    f, t, mic_stft = scipy.signal.stft(data_mic, fs=sr_mic, nperseg=2048,)
    fmax = 800
    axs[0].plot(t_mic, data_mic)
    axs[1].pcolormesh(t, f, np.abs(mic_stft))
    axs[1].set_ylim([0, fmax])

def visual_mic_ref(data_mic1, data_mic2, axs):
    scale = np.max(np.abs(data_mic1)) / np.max(np.abs(data_mic2))
    data_mic2 = data_mic2 * scale
    t_mic_1 = np.arange(len(data_mic1)) / sr_mic
    t_mic_2 = np.arange(len(data_mic2)) / sr_mic

    axs[0].plot(t_mic_1, data_mic1)
    axs[0].plot(t_mic_2, data_mic2)
def load_data(dir, save=False):
    save_dir = dir + '_processed'
    files = os.listdir(dir)
    files_imu = [f for f in files if f.split('_')[0] == 'IMU']
    files_mic = [f for f in files if f.split('_')[0] == 'MIC']
    references = [f for f in files if f.split('_')[0] == 'Steth']
    number_of_files = len(files_imu)
    for i in range(2, number_of_files):
        imu = os.path.join(dir, files_imu[i])
        mic = os.path.join(dir, files_mic[i])
        reference = os.path.join(dir, references[i])

        data_mic, sr = librosa.load(mic, sr=sr_mic)
        data_reference, sr = librosa.load(reference, sr=sr_mic)
        data_imu = np.loadtxt(imu, delimiter=',', skiprows=1, usecols=(0, 1), converters={1:converter}) # only load Y and timestamp
        data_imu = IMU_resample(data_imu)

        # data_imu = scipy.signal.filtfilt(*filter_list['imu'], data_imu, axis=0)
        # data_mic = scipy.signal.filtfilt(*filter_list['heartbeat filter'], data_mic)
        # data_reference = scipy.signal.filtfilt(*filter_list['heartbeat filter'], data_reference)

        data_mic, data_imu = synchronize_playback(data_mic, data_imu, data_reference,)

        fig, axs = plt.subplots(4, 2)
        visual_imu(data_imu, axs[0])
        visual_mic(data_mic, axs[1])
        visual_mic(data_reference, axs[2])
        visual_mic_ref(data_mic, data_reference, axs[3])
        # transfer_function(data_mic, data_reference, axs[3])
        plt.show()
        # break
        
if __name__ == "__main__":
    load_data('test', save=False)
