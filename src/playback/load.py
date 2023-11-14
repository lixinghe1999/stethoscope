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
               'imu': scipy.signal.butter(4, [1, 100], 'bandpass', fs=sr_imu)}
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

def load_measurement(dir):
    files = os.listdir(dir)
    files_imu = [f for f in files if f.split('_')[0] == 'IMU']
    files_mic = [f for f in files if f.split('_')[0] == 'MIC']
    number_of_files = len(files_imu)
    for i in range(0, number_of_files):
        imu = os.path.join(dir, files_imu[i])
        mic = os.path.join(dir, files_mic[i])
        
        data_mic, sr = librosa.load(mic, sr=sr_mic)
        data_imu = np.loadtxt(imu, delimiter=',', skiprows=1, usecols=(2, 4), converters={4:converter}) # only load Y and timestamp

        data_imu = synchronization_one(data_imu)
        data_imu = scipy.signal.filtfilt(*filter_list['imu'], data_imu, axis=0)
        data_mic = scipy.signal.filtfilt(*filter_list['heartbeat filtered'], data_mic)
        fig, axs = plt.subplots(2, 2)

        heartbeat_imu, data_mic = process_playback(data_imu, data_mic)
        visual_imu(data_imu, heartbeat_imu, axs[0])
        visual_mic(data_mic, axs[1])
        plt.show()
def load_data(dir, save=False):
    metric = metrics.AudioMetrics(rate=44100)
    files = os.listdir(dir)
    files_imu = [f for f in files if f.split('_')[0] == 'IMU']
    files_mic = [f for f in files if f.split('_')[0] == 'MIC']
    references = np.loadtxt(os.path.join(dir, 'reference.txt'), dtype=str)
    number_of_files = len(files_imu)
    for i in range(80, number_of_files):
        imu = os.path.join(dir, files_imu[i])
        mic = os.path.join(dir, files_mic[i])
        reference = references[i][0]

        data_mic, sr = librosa.load(mic, sr=sr_mic)
        data_reference, sr = librosa.load(reference, sr=sr_mic)
        data_imu = np.loadtxt(imu, delimiter=',', skiprows=1, usecols=(2, 4), converters={4:converter}) # only load Y and timestamp
        data_imu = IMU_resample(data_imu)
        data_mic, data_imu = synchronize_playback(data_mic, data_imu, data_reference,)
        
        print(data_mic.shape, data_reference.shape, data_imu.shape)
        if save:
            scipy.io.wavfile.write(mic, sr_mic, data_mic)
            scipy.io.wavfile.write(imu.replace('csv', 'wav'), sr_imu, data_imu)
        # data_imu = scipy.signal.filtfilt(*filter_list['imu'], data_imu, axis=0)
        # data_mic = scipy.signal.filtfilt(*filter_list['heartbeat filtered'], data_mic)
        # data_reference = scipy.signal.filtfilt(*filter_list['heartbeat filtered'], data_reference)
        # fig, axs = plt.subplots(3, 2)

        # heartbeat_imu, data_mic = process_playback(data_imu, data_mic)
        # visual_imu(data_imu, heartbeat_imu, axs[0])
        # visual_mic(data_mic, axs[1])
        # visual_mic(data_reference, axs[2])
        # plt.show()
        # break
        
if __name__ == "__main__":
    load_data('smartphone/CHSC/set_a')
    # load_measurement('smartphone/measurement')