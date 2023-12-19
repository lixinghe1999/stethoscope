import datetime
import numpy as np
import scipy
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

    else:
        sync_clip = int(4 * sr_mic)
        offset = 0
    # pre_rmse = np.sqrt(np.mean((record[offset:sync_clip] - playback[offset:sync_clip])**2))

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
    if  len(record) != len(playback):
        record[:offset] *= 0
        playback[:offset] *= 0
        # record = record[offset:]
        # playback = playback[offset:] 
        if imu is not None:
            offset = int(1 * sr_imu)
            imu[:offset] = 0
    if len(record) < len(playback):
        record = np.pad(record, (0, len(playback) - len(record)), 'constant')
    else:
        record = record[:len(playback)]
    
    # pre_cos_sim = abs(np.dot(envelop_record[offset:], envelop_playback[offset:]) / (np.linalg.norm(envelop_record[offset:]) * np.linalg.norm(envelop_playback[offset:])))
    # envelop_record = np.abs(scipy.signal.hilbert(record[:sync_clip]))
    # envelop_playback = np.abs(scipy.signal.hilbert(playback[:sync_clip]))
    # post_cos_sim = abs(np.dot(envelop_record, envelop_playback) / (np.linalg.norm(envelop_record) * np.linalg.norm(envelop_playback)))
    # post_rmse = np.sqrt(np.mean((record[:sync_clip] - playback[:sync_clip])**2))
    return record, playback, imu, [0, 0, 0, 0]
    # return record, playback, imu, [pre_cos_sim, post_cos_sim, pre_rmse, post_rmse]

def converter(x):
    time_str = x.decode("utf-8")
    time_str = time_str.split('_')
    time_str = '.'.join(time_str) 
    x = (datetime.datetime.strptime(time_str, '%S.%f') - datetime.datetime(1900, 1, 1)).total_seconds()
    return x

def IMU_resample(data_imu):
    data_imu, time_imu = data_imu[:, 0], data_imu[:, -1]
    time_revise = time_imu - time_imu[0]
    time_revise[time_revise < 0] += 60
    length = (time_revise[-1] - time_revise[0]) 
    f_imu = scipy.interpolate.interp1d(time_revise, data_imu, axis=0)
    time_imu = np.arange(0, length, 1/sr_imu)
    data_imu = f_imu(time_imu)
    return data_imu
