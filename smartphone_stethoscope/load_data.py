import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy
import filter
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

def visualize(data_imu, data_mic, data_ppg, data_scian=None):
    data_mic, heartbeat_imu, heartbeat_ppg = denoise(data_imu, data_mic, data_ppg)
    fig, axs = plt.subplots(4, 2)
    f, t, imu_stft = scipy.signal.stft(data_imu, axis=0, fs=sr_imu, nperseg=128,)
    t_imu = np.arange(len(data_imu)) / sr_imu
    axs[0, 0].plot(t_imu, data_imu)

    axs[0, 1].pcolormesh(t, f, np.abs(imu_stft))
    axs[0, 0].plot(np.array(heartbeat_imu)/ sr_imu, data_imu[heartbeat_imu], "x")
    
    t_ppg = np.arange(len(data_ppg)) / sr_ppg
    axs[1, 0].plot(t_ppg, data_ppg)
    axs[1, 0].plot(np.array(heartbeat_ppg)/ sr_ppg, data_ppg[heartbeat_ppg], "x")

    axs[1, 1].scatter(np.array(heartbeat_imu)/sr_imu, np.ones_like(heartbeat_imu), c='r')
    axs[1, 1].scatter(np.array(heartbeat_ppg)/sr_ppg, np.ones_like(heartbeat_ppg)*0, c='b')
    beat_diff = np.tile(np.array(heartbeat_imu)/sr_imu, (len(heartbeat_ppg), 1)) \
    - np.tile(np.array(heartbeat_ppg)/sr_ppg, (len(heartbeat_imu), 1)).T
    match = np.argmin(np.abs(beat_diff), axis=1)
    min_beat_diff = np.min(np.abs(beat_diff), axis=1)
    print('pulse difference', min_beat_diff.mean(), min_beat_diff.std())
    for i, m in enumerate(match):
        axs[1, 1].plot([heartbeat_imu[m]/sr_imu, heartbeat_ppg[i]/sr_ppg], [1, 0], c='g')

    t_mic = np.arange(len(data_mic)) / sr_mic
    f, t, mic_stft = scipy.signal.stft(data_mic, fs=sr_mic, nperseg=2048,)
    fmax = 400
    axs[2, 0].plot(t_mic, data_mic)
    axs[2, 1].pcolormesh(t, f, np.abs(mic_stft))
    axs[2, 1].set_ylim([0, fmax])
    if data_scian is not None:
        t_mic = np.arange(len(data_scian)) / sr_mic
        f, t, scian_stft = scipy.signal.stft(data_scian, fs=sr_mic, nperseg=2048,)
        fmax = 400
        axs[3, 0].plot(t_mic, data_scian)
        axs[3, 1].pcolormesh(t, f, np.abs(scian_stft))
        axs[3, 1].set_ylim([0, fmax])

    plt.show()

def load_data_wo_ref(dir):
    files = os.listdir(dir)
    files_imu = [f for f in files if f.split('_')[0] == 'IMU']
    files_mic = [f for f in files if f.split('_')[0] == 'MIC']
    files_ppg = [f for f in files if f.split('_')[0] == 'PPG']
    assert len(files_imu) == len(files_mic) == len(files_ppg)
    number_of_files = len(files_imu)
    for i in range(0, number_of_files):
        imu = os.path.join(dir, files_imu[i])
        mic = os.path.join(dir, files_mic[i])
        ppg = os.path.join(dir, files_ppg[i])

        data_mic, sr = librosa.load(mic, sr=sr_mic)

        data_imu = np.loadtxt(imu, delimiter=',', skiprows=1, usecols=(2, 4), converters={4:converter}) # only load Y and timestamp
        data_ppg = np.loadtxt(ppg, delimiter=',', usecols=(0, 1), converters={1:converter})

        data_imu, data_ppg = synchronization(data_imu, data_ppg,)

        data_imu = scipy.signal.filtfilt(*filter.filter_list['imu'], data_imu, axis=0)
        data_mic = scipy.signal.filtfilt(*filter.filter_list['heartbeat filtered'], data_mic)
        visualize(data_imu, data_mic, data_ppg)
def load_data(dir):
    metric = metrics.AudioMetrics(rate=44100)
    files = os.listdir(dir)
    files_imu = [f for f in files if f.split('_')[0] == 'IMU']
    files_mic = [f for f in files if f.split('_')[0] == 'MIC']
    files_ppg = [f for f in files if f.split('_')[0] == 'PPG']
    files_scian = [f for f in files if f.split('_')[0] == 'SCIAN']
    assert len(files_imu) == len(files_mic) == len(files_ppg)
    number_of_files = len(files_imu)
    for i in range(0, number_of_files):
        imu = os.path.join(dir, files_imu[i])
        mic = os.path.join(dir, files_mic[i])
        ppg = os.path.join(dir, files_ppg[i])
        scian = os.path.join(dir, files_scian[i]) if len(files_scian) > 0 else None

        data_mic, sr = librosa.load(mic, sr=sr_mic)
        data_scian, sr = librosa.load(scian, sr=sr_mic) if scian is not None else (None, None)
        
        mic_drift = int(drift_parse(files_mic[i], files_scian[i]) * sr_mic)
        if mic_drift > 0:
            data_mic = data_mic[mic_drift:mic_drift + len(data_scian)]
        else:
            data_mic = np.pad(data_mic, (abs(mic_drift), 0))
            data_mic = data_mic[:len(data_scian)]
        print(metric.evaluation(data_scian, data_mic))

        data_imu = np.loadtxt(imu, delimiter=',', skiprows=1, usecols=(2, 4), converters={4:converter}) # only load Y and timestamp
        data_ppg = np.loadtxt(ppg, delimiter=',', usecols=(0, 1), converters={1:converter})

        data_imu, data_ppg = synchronization(data_imu, data_ppg,)

        data_imu = scipy.signal.filtfilt(*filter.filter_list['imu'], data_imu, axis=0)
        data_mic = scipy.signal.filtfilt(*filter.filter_list['heartbeat filtered'], data_mic)
        data_scian = scipy.signal.filtfilt(*filter.filter_list['heartbeat filtered'], data_scian) if scian is not None else None
        visualize(data_imu, data_mic, data_ppg, data_scian)
        # break
        
if __name__ == "__main__":
    load_data_wo_ref('dataset')
    load_data('dataset_scian')