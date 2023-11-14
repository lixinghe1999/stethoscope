import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy
import sys
sys.path.append('../')
from utils import *
import metrics

import warnings
warnings.filterwarnings("ignore")
filter_list = {'heartbeat pure': scipy.signal.butter(4, 400, 'lowpass', fs=sr_mic),
               'heartbeat filtered': scipy.signal.butter(4, [20, 1000], 'bandpass', fs=sr_mic),
               'imu': scipy.signal.butter(4, [1, 25], 'bandpass', fs=sr_imu)}
sr_mic = 44100
sr_imu = 400
sr_ppg = 25

def visual_imu(data_imu, heartbeat_imu, axs):
    '''
    two axes to show IMU data in waveform and STFT
    '''
    f, t, imu_stft = scipy.signal.stft(data_imu, axis=0, fs=sr_imu, nperseg=128,)
    t_imu = np.arange(len(data_imu)) / sr_imu
    axs[0].plot(t_imu, data_imu)
    axs[0].plot(np.array(heartbeat_imu)/ sr_imu, data_imu[heartbeat_imu], "x")
    axs[1].pcolormesh(t, f, np.abs(imu_stft))
def visual_ppg(data_ppg, heartbeat_imu, heartbeat_ppg, axs):
    '''
    two axes to show PPG data and detected heartbeat from IMU and PPG
    '''
    t_ppg = np.arange(len(data_ppg)) / sr_ppg
    axs[0].plot(t_ppg, data_ppg)
    axs[0].plot(np.array(heartbeat_ppg)/ sr_ppg, data_ppg[heartbeat_ppg], "x")

    axs[1].scatter(np.array(heartbeat_imu)/sr_imu, np.ones_like(heartbeat_imu), c='r')
    axs[1].scatter(np.array(heartbeat_ppg)/sr_ppg, np.ones_like(heartbeat_ppg)*0, c='b')
    beat_diff = np.tile(np.array(heartbeat_imu)/sr_imu, (len(heartbeat_ppg), 1)) \
    - np.tile(np.array(heartbeat_ppg)/sr_ppg, (len(heartbeat_imu), 1)).T
    match = np.argmin(np.abs(beat_diff), axis=1)
    min_beat_diff = np.min(np.abs(beat_diff), axis=1)
    for i, m in enumerate(match):
        axs[1].plot([heartbeat_imu[m]/sr_imu, heartbeat_ppg[i]/sr_ppg], [1, 0], c='g')
    print('pulse difference', min_beat_diff.mean(), min_beat_diff.std())
def visual_mic(data_mic, axs):
    '''
    two axes to show microphone data in waveform and STFT, apply to smartphone or stethoscope recordings
    '''
    t_mic = np.arange(len(data_mic)) / sr_mic
    f, t, mic_stft = scipy.signal.stft(data_mic, fs=sr_mic, nperseg=2048,)
    fmax = 400
    axs[0].plot(t_mic, data_mic)
    axs[1].pcolormesh(t, f, np.abs(mic_stft))
    axs[1].set_ylim([0, fmax])


def load_data(dir):
    files = os.listdir(dir)
    files_imu = [f for f in files if f.split('_')[0] == 'IMU']
    files_mic = [f for f in files if f.split('_')[0] == 'MIC']
    files_ppg = [f for f in files if f.split('_')[0] == 'PPG']
    number_of_files = len(files_imu)
    for i in range(0, number_of_files):
        imu = os.path.join(dir, files_imu[i])
        mic = os.path.join(dir, files_mic[i])
        ppg = os.path.join(dir, files_ppg[i])

        data_mic, sr = librosa.load(mic, sr=sr_mic)

        data_imu = np.loadtxt(imu, delimiter=',', skiprows=1, usecols=(2, 4), converters={4:converter}) # only load Y and timestamp
        data_ppg = np.loadtxt(ppg, delimiter=',', usecols=(0, 1), converters={1:converter})
        data_imu, data_ppg = synchronization_two(data_imu, data_ppg,)

        data_imu = scipy.signal.filtfilt(*filter_list['imu'], data_imu, axis=0)
        data_mic = scipy.signal.filtfilt(*filter_list['heartbeat filtered'], data_mic)
        
        heartbeat_imu, data_mic, heartbeat_ppg = process_experiment(data_imu, data_mic, data_ppg)

        fig, axs = plt.subplots(3, 2)
        visual_imu(data_imu, heartbeat_imu, axs[0])
        visual_ppg(data_ppg, heartbeat_imu, heartbeat_ppg, axs[1])
        visual_mic(data_mic, axs[2])
        plt.show()
        # break
        
if __name__ == "__main__":
    load_data('scian')