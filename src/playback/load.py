import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import scipy
import sys
from tqdm import tqdm
sys.path.append('../')
from utils import *

import warnings
warnings.filterwarnings("ignore")
filter_list = {'heartbeat pure': scipy.signal.butter(4, 400, 'lowpass', fs=sr_mic),
               'heartbeat filtered': scipy.signal.butter(4, [20, 1000], 'bandpass', fs=sr_mic),
               'imu': scipy.signal.butter(4, [1, 100], 'bandpass', fs=sr_imu)}
sr_mic = 4000
sr_imu = 400
sr_ppg = 25

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
    t_mic = np.arange(len(data_mic1)) / sr_mic
    axs[0].plot(t_mic, data_mic1)
    axs[0].plot(t_mic, data_mic2)
def load_data(dir, save=False):
    save_dir = dir + '_processed'
    files = os.listdir(dir)
    files_imu = [f for f in files if f.split('_')[0] == 'IMU']
    files_mic = [f for f in files if f.split('_')[0] == 'MIC']
    references = np.loadtxt(os.path.join(dir, 'reference.txt'), dtype=str)
    f = open(os.path.join(save_dir, 'reference.txt'), 'w')
    number_of_files = len(files_imu)
    for i in tqdm(range(0, number_of_files)):
        imu = os.path.join(dir, files_imu[i])
        mic = os.path.join(dir, files_mic[i])
        reference = references[i][0]
        f.write(files_imu[i].replace('csv', 'flac') + ' ' + files_mic[i].replace('wav', 'flac') + ' ' + references[i][-1] + '\n')
        data_mic, sr = librosa.load(mic, sr=sr_mic)
        data_reference, sr = librosa.load(reference, sr=sr_mic)
        data_imu = np.loadtxt(imu, delimiter=',', skiprows=1, usecols=(0, 1), converters={1:converter}) # only load Y and timestamp
        data_imu = IMU_resample(data_imu)

        data_mic, data_imu = synchronize_playback(data_mic, data_imu, data_reference,)
        data_imu = scipy.signal.filtfilt(*filter_list['imu'], data_imu)
        data_mic = scipy.signal.filtfilt(*filter_list['heartbeat filtered'], data_mic)
        data_mic /= np.max(np.abs(data_mic))
        data_reference = scipy.signal.filtfilt(*filter_list['heartbeat filtered'], data_reference)
        data_reference /= np.max(np.abs(data_reference))
        if save:
            data = np.stack((data_mic, data_reference), axis=1)
            sf.write(os.path.join(save_dir, files_mic[i]).replace('wav', 'flac'), data, sr_mic)
            sf.write(os.path.join(save_dir, files_imu[i]).replace('csv', 'flac'), data_imu, sr_imu)
        else:
            fig, axs = plt.subplots(4, 2)

            visual_imu(data_imu, axs[0])
            visual_mic(data_mic, axs[1])
            visual_mic(data_reference, axs[2])
            visual_mic_ref(data_mic, data_reference, axs[3])
            fig.delaxes(axs[3, 1]) 
            plt.show()
    f.close()  
if __name__ == "__main__":
    load_data('smartphone/CHSC/set_a', save=True)
