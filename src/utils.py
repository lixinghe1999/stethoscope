import datetime
import numpy as np
import scipy
import heartbeat_segment
import ppg_pipeline
import matplotlib.pyplot as plt
sr_mic = 4000
sr_imu = 400
sr_ppg = 25
def synchronize_playback(record, playback, imu=None):
    '''
    record has left and right offset compared to playback
    '''
    if len(record) != len(playback):
        sync_clip = int(2 * sr_mic)
        offset = int(1 * sr_mic)
        record = record[offset:]
        playback = playback[offset:] 
        if imu is not None:
            offset = int(1 * sr_imu)
            imu = imu[offset:]
    else:
        sync_clip = int(8 * sr_mic)
        offset = 0
    pre_rmse = np.sqrt(np.mean((record[offset:sync_clip] - playback[offset:sync_clip])**2))

    envelop_record = np.abs(scipy.signal.hilbert(record))[:sync_clip]
    envelop_playback = np.abs(scipy.signal.hilbert(playback))[:sync_clip]
    correlation = np.correlate(envelop_record, envelop_playback, mode='full')

    shift = np.argmax(correlation) - sync_clip
    record = np.roll(record, -shift)

    if imu is not None:
        shift = int(shift * sr_imu / sr_mic)
        imu = np.roll(imu, -shift)
        expect_len = int(len(playback) * sr_imu / sr_mic)
        if len(imu) < expect_len:
            imu = np.pad(imu, (0, expect_len - len(imu)), 'constant')
        else:
            imu = imu[:expect_len]
    if len(record) < len(playback):
        record = np.pad(record, (0, len(playback) - len(record)), 'constant')
    else:
        record = record[:len(playback)]    
    pre_cos_sim = abs(np.dot(envelop_record[offset:], envelop_playback[offset:]) / (np.linalg.norm(envelop_record[offset:]) * np.linalg.norm(envelop_playback[offset:])))
    envelop_record = np.abs(scipy.signal.hilbert(record[:sync_clip - offset]))
    envelop_playback = np.abs(scipy.signal.hilbert(playback[:sync_clip - offset]))
    post_cos_sim = abs(np.dot(envelop_record, envelop_playback) / (np.linalg.norm(envelop_record) * np.linalg.norm(envelop_playback)))
    post_rmse = np.sqrt(np.mean((record[:sync_clip - offset] - playback[:sync_clip - offset])**2))
    if imu is not None:
        imu /= np.max(np.abs(imu))
    record = record / np.max(np.abs(record))
    playback = playback / np.max(np.abs(playback))
    # return record, playback, imu, [0, 0, 0, 0]
    return record, playback, imu, [pre_cos_sim, post_cos_sim, pre_rmse, post_rmse]

def converter(x):
    time_str = x.decode("utf-8")
    time_str = time_str.split('_')
    time_str = '.'.join(time_str) 
    x = (datetime.datetime.strptime(time_str, '%S.%f') - datetime.datetime(1900, 1, 1)).total_seconds()
    return x

def revise_timestampe(data, timestamps):
    unique_timestamps, unique_indices = np.unique(timestamps, return_index=True)
    # Sort the unique timestamps and their corresponding indices
    sorted_indices = np.argsort(unique_timestamps)
    sorted_timestamps = unique_timestamps[sorted_indices]
    sorted_values = data[unique_indices[sorted_indices]]
    return sorted_values, sorted_timestamps

def IMU_resample(data_imu):
    data_imu, time_imu = data_imu[:, 0], data_imu[:, -1]
    real_sr_imu = time_imu.shape[0]/ (time_imu[-1] - time_imu[0]) 
    print('real imu sr:', real_sr_imu, 'target:', sr_imu)
    data_imu, time_imu = revise_timestampe(data_imu, time_imu)
    f_imu = scipy.interpolate.interp1d(time_imu - time_imu[0], data_imu, axis=0)
    print(time_imu[-1]- time_imu[0])
    time_imu = np.arange(0, time_imu[-1] - time_imu[0], 1/sr_imu)
    data_imu = f_imu(time_imu)
   
    return data_imu

def process_experiment(data_imu, data_mic, data_ppg):
    heartbeat_imu = heartbeat_segment.heart_rate_estimation(data_imu, plot=False)
    heartbeat_ppg = ppg_pipeline.pipeline(data_ppg)

    return heartbeat_imu, data_mic, heartbeat_ppg
def process_playback(data_imu, data_mic,):
    heartbeat_imu = heartbeat_segment.heart_rate_estimation(data_imu, plot=False)

    return heartbeat_imu, data_mic