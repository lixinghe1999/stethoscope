import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy
import filter
import heartbeat_segment
import ppg_pipeline

import warnings
warnings.filterwarnings("ignore")
import datetime
def converter(x):
    time_str = x.decode("utf-8")
    time_str = '.'.join(time_str.split('_')[1:]) # remove date
    x = (datetime.datetime.strptime(time_str, '%H%M%S.%f') - datetime.datetime(1900, 1, 1)).total_seconds()
    return x
sr_mic = 48000
sr_imu = 400
sr_ppg = 25
def visualize(data_imu, data_mic, data_ppg):
    fig, axs = plt.subplots(5, 2)
    heartbeat_imu = heartbeat_segment.heart_rate_estimation(data_imu[:, 1], plot=True)
    f, t, imu_stft = scipy.signal.stft(data_imu, axis=0, fs=sr_imu, nperseg=128,)
    t_imu = np.arange(len(data_imu)) / sr_imu
    axs[0, 0].plot(t_imu, data_imu[:, 0])
    axs[1, 0].plot(t_imu, data_imu[:, 1])
    axs[2, 0].plot(t_imu, data_imu[:, 2])

    axs[0, 1].pcolormesh(t, f, np.abs(imu_stft[:, 0]))
    axs[1, 1].pcolormesh(t, f, np.abs(imu_stft[:, 1]))
    axs[2, 1].pcolormesh(t, f, np.abs(imu_stft[:, 2])) 
    
    axs[1, 0].plot(np.array(heartbeat_imu)/ sr_imu, data_imu[heartbeat_imu, 1], "x")
    
    t_mic = np.arange(len(data_mic)) / sr_mic
    f, t, mic_stft = scipy.signal.stft(data_mic, fs=sr_mic, nperseg=2048,)
    fmax = 400
    axs[3, 0].plot(t_mic, data_mic)
    axs[3, 1].pcolormesh(t, f, np.abs(mic_stft))
    axs[3, 1].set_ylim([0, fmax])

    raw_ppg, data_ppg, heartbeat_ppg = ppg_pipeline.pipeline(data_ppg)
    
    t_ppg = np.arange(len(data_ppg)) / sr_ppg
    axs[4, 0].plot(t_ppg, raw_ppg)
    axs[4, 0].plot(np.array(heartbeat_ppg)/ sr_ppg, raw_ppg[heartbeat_ppg], "x")

    axs[4, 1].scatter(np.array(heartbeat_imu)/sr_imu, np.ones_like(heartbeat_imu), c='r')
    axs[4, 1].scatter(np.array(heartbeat_ppg)/sr_ppg, np.ones_like(heartbeat_ppg)*0, c='b')
    beat_diff = np.tile(np.array(heartbeat_imu)/sr_imu, (len(heartbeat_ppg), 1)) \
    - np.tile(np.array(heartbeat_ppg)/sr_ppg, (len(heartbeat_imu), 1)).T
    match = np.argmin(np.abs(beat_diff), axis=1)
    min_beat_diff = np.min(np.abs(beat_diff), axis=1)
    print('pulse difference', min_beat_diff.mean(), min_beat_diff.std())
    for i, m in enumerate(match):
        axs[4, 1].plot([heartbeat_imu[m]/sr_imu, heartbeat_ppg[i]/sr_ppg], [1, 0], c='g')
    plt.show()
def revise_timestampe(data, timestamps):
    unique_timestamps, unique_indices = np.unique(timestamps, return_index=True)

    # Sort the unique timestamps and their corresponding indices
    sorted_indices = np.argsort(unique_timestamps)
    sorted_timestamps = unique_timestamps[sorted_indices]
    sorted_values = data[unique_indices[sorted_indices], :]
    return sorted_values, sorted_timestamps
def synchronization(data_imu, data_ppg):
    data_imu, time_imu = data_imu[:, :-1], data_imu[:, -1]
    data_ppg, time_ppg = data_ppg[:, 0], data_ppg[:, -1]
    sensor_drift = np.argmin(abs(time_imu[0] - time_ppg))
    data_ppg = data_ppg[sensor_drift:]; time_ppg = time_ppg[sensor_drift:]
    real_sr_imu = time_imu.shape[0]/ (time_imu[-1] - time_imu[0]) 
    real_sr_ppg = time_ppg.shape[0]/ (time_ppg[-1] - time_ppg[0])  
    print('real sample rate:', real_sr_imu, real_sr_ppg)

    data_imu, time_imu = revise_timestampe(data_imu, time_imu)
    f_imu = scipy.interpolate.interp1d(time_imu - time_imu[0], data_imu, axis=0)
    time_imu = np.arange(0, time_imu[-1] - time_imu[0], 1/sr_imu)
    data_imu = f_imu(time_imu)
    data_imu = np.nan_to_num(data_imu)

    f_ppg = scipy.interpolate.interp1d(time_ppg - time_ppg[0], data_ppg, axis=0)
    time_ppg = np.arange(0, time_ppg[-1] - time_ppg[0], 1/sr_ppg)
    data_ppg = f_ppg(time_ppg)
    return data_imu, data_ppg
def load_data(dir):
    files = os.listdir(dir)
    modalitlies = 3
    number_recordings = len(files) // modalitlies
    print("we are loading dataset of IMU, microphone and PPG")
    files_imu = files[:number_recordings]; files_mic = files[number_recordings:2*number_recordings]; files_ppg = files[2*number_recordings:]

    for i in range(1, number_recordings):
        imu = os.path.join(dir, files_imu[i])
        mic = os.path.join(dir, files_mic[i])
        ppg = os.path.join(dir, files_ppg[i])

        data_mic, sr = librosa.load(mic, sr=sr_mic)

        data_imu = np.loadtxt(imu, delimiter=',', skiprows=1, usecols=(1,2,3,4), converters={4:converter})
        data_ppg = np.loadtxt(ppg, delimiter=',', usecols=(0, 1), converters={1:converter})

        data_imu, data_ppg = synchronization(data_imu, data_ppg)

        data_imu = scipy.signal.filtfilt(*filter.filter_list['imu'], data_imu, axis=0)
        data_mic = scipy.signal.filtfilt(*filter.filter_list['heartbeat pure'], data_mic)

        # heartbeat = (sr_mic / sr_imu) * heartbeat   
        # interval = 8000
        # segmentation = np.stack([heartbeat - interval//2, heartbeat + interval//2])
        # for i in range(len(heartbeat)):
        #     plt.plot(data_mic[int(segmentation[0, i]):int(segmentation[1, i])])
        # plt.show()

        visualize(data_imu, data_mic, data_ppg)
        # break
        
if __name__ == "__main__":
    load_data('dataset')