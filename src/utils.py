import datetime
import numpy as np
import scipy
import heartbeat_segment
import ppg_pipeline
sr_mic = 4000
sr_imu = 400
sr_ppg = 25
def synchronize_playback(record, imu, playback, flag=False):
    '''
    record has left and right offset compared to playback
    '''
    assert len(record) > len(playback)
    envelop_record = np.abs(scipy.signal.hilbert(record))
    envelop_playback = np.abs(scipy.signal.hilbert(playback))
    if flag:
        envelop_record = envelop_record[:1 * sr_mic]
        envelop_playback = envelop_playback[:1 * sr_mic + len(playback) - len(record)]
    correlation = np.correlate(envelop_record, envelop_playback, mode='valid')
    shift = np.argmax(correlation) + 1 * sr_mic
    shift_imu = int(shift * sr_imu / sr_mic)
    imu_length = int(len(playback) * sr_imu / sr_mic)
    return record[shift: shift + len(playback)], imu[shift_imu: shift_imu + imu_length]

def converter(x):
    time_str = x.decode("utf-8")
    time_str = time_str.split('_')
    if len(time_str) == 3:
        time_str = time_str[1:] # remove date
        time_str = '.'.join(time_str) 
        x = (datetime.datetime.strptime(time_str, '%H%M%S.%f') - datetime.datetime(1900, 1, 1)).total_seconds()
    else:
        time_str = '.'.join(time_str) 
        x = (datetime.datetime.strptime(time_str, '%S.%f') - datetime.datetime(1900, 1, 1)).total_seconds()
    return x
def drift_parse(t1, t2):
    t1 = '.'.join(t1.split('_')[2:])[:-4] # remove date
    t1 = datetime.datetime.strptime(t1, '%H%M%S.%f')
    t2 = '.'.join(t2.split('_')[2:])[:-4] # remove date
    t2 = datetime.datetime.strptime(t2, '%H%M%S.%f')
    return (t2 - t1).total_seconds()
def revise_timestampe(data, timestamps):
    unique_timestamps, unique_indices = np.unique(timestamps, return_index=True)
    # Sort the unique timestamps and their corresponding indices
    sorted_indices = np.argsort(unique_timestamps)
    sorted_timestamps = unique_timestamps[sorted_indices]
    sorted_values = data[unique_indices[sorted_indices]]
    return sorted_values, sorted_timestamps

def IMU_resample(data_imu, ):
    data_imu, time_imu = data_imu[:, 0], data_imu[:, -1]
    real_sr_imu = time_imu.shape[0]/ (time_imu[-1] - time_imu[0]) 

    data_imu, time_imu = revise_timestampe(data_imu, time_imu)
    f_imu = scipy.interpolate.interp1d(time_imu - time_imu[0], data_imu, axis=0)
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