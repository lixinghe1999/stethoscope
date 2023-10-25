import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy
import filter
import heartbeat_segment
import warnings
warnings.filterwarnings("ignore")
def converter(x):
    if type(x) == str:
        x = x[-6:].replace('_', '.')
    return float(x)
sr_mic = 48000
sr_imu = 400
def visualize(data_imu, data_mic):
    t_imu = np.arange(len(data_imu)) / sr_imu
    t_mic = np.arange(len(data_mic)) / sr_mic

    fig, axs = plt.subplots(4, 2)
    f, t, imu_stft = scipy.signal.stft(data_imu, axis=0, fs=sr_imu, nperseg=128,)

    axs[0, 0].plot(t_imu, data_imu[:, 0])
    axs[1, 0].plot(t_imu, data_imu[:, 1])
    axs[2, 0].plot(t_imu, data_imu[:, 2])

    axs[0, 1].pcolormesh(t, f, np.abs(imu_stft[:, 0]))
    axs[1, 1].pcolormesh(t, f, np.abs(imu_stft[:, 1]))
    axs[2, 1].pcolormesh(t, f, np.abs(imu_stft[:, 2])) 

    f, t, mic_stft = scipy.signal.stft(data_mic, fs=sr_mic, nperseg=2048,)
    fmax = 400
    axs[3, 0].plot(t_mic, data_mic)
    axs[3, 1].pcolormesh(t, f, np.abs(mic_stft))
    axs[3, 1].set_ylim([0, fmax])
    plt.show()
def load_data(dir):
    dir_imu = os.path.join(dir, 'imu')
    dir_mic = os.path.join(dir, 'mic')
    files_imu = os.listdir(dir_imu); files_mic = os.listdir(dir_mic)

    for i in range(len(files_imu)):
        imu = os.path.join(dir_imu, files_imu[i])
        mic = os.path.join(dir_mic, files_mic[i])

        data_imu = np.loadtxt(imu, delimiter=',', skiprows=1, usecols=(1,2,3,))
        data_mic, sr = librosa.load(mic, sr=None)
        
        data_imu = scipy.signal.filtfilt(*filter.filter_list['imu'], data_imu, axis=0)
        data_mic = scipy.signal.filtfilt(*filter.filter_list['heartbeat pure'], data_mic)

        # heartbeat = heartbeat_segment.heart_rate_estimation(data_imu[:, 1])
        # HRV = len(heartbeat) / (len(data_imu) / sr_imu) * 60
        # print(HRV)

        # heartbeat = (sr_mic / sr_imu) * heartbeat   
        # interval = 8000
        # segmentation = np.stack([heartbeat - interval//2, heartbeat + interval//2])
        # for i in range(len(heartbeat)):
        #     plt.plot(data_mic[int(segmentation[0, i]):int(segmentation[1, i])])
        # plt.show()

        visualize(data_imu, data_mic)
        break
        
if __name__ == "__main__":
    load_data('dataset')