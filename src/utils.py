import datetime
import numpy as np
import scipy
import wtdenoise
import heartbeat_segment
import ppg_pipeline
sr_mic = 44100
sr_imu = 400
sr_ppg = 25
def synchronize_playback(record, imu, playback):
    '''
    record has left and right offset compared to playback
    '''
    assert len(record) > len(playback); print('the record is shorter than playback')
    correlation = np.correlate(record, playback, mode='valid')
    shift = np.argmax(correlation)
    right_pad = len(record) - shift - len(playback)
    print('shift:', shift, 'right_pad:', right_pad)
    shift_imu = int(shift * sr_imu / sr_mic)
    right_pad_imu = int(right_pad * sr_imu / sr_mic)
    return record[shift: -right_pad], imu[shift_imu: -right_pad_imu]
def converter(x):
    time_str = x.decode("utf-8")
    time_str = '.'.join(time_str.split('_')[1:]) # remove date
    x = (datetime.datetime.strptime(time_str, '%H%M%S.%f') - datetime.datetime(1900, 1, 1)).total_seconds()
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
def synchronization_two(data_imu, data_ppg):
    data_imu, time_imu = data_imu[:, 0], data_imu[:, -1]
    data_ppg, time_ppg = data_ppg[:, 0], data_ppg[:, -1]
    sensor_drift = np.argmin(abs(time_imu[0] - time_ppg))
    data_ppg = data_ppg[sensor_drift:]; time_ppg = time_ppg[sensor_drift:]
    real_sr_imu = time_imu.shape[0]/ (time_imu[-1] - time_imu[0]) 
    real_sr_ppg = time_ppg.shape[0]/ (time_ppg[-1] - time_ppg[0])  
    # print('real sample rate:', real_sr_imu, real_sr_ppg)

    data_imu, time_imu = revise_timestampe(data_imu, time_imu)
    f_imu = scipy.interpolate.interp1d(time_imu - time_imu[0], data_imu, axis=0)
    time_imu = np.arange(0, time_imu[-1] - time_imu[0], 1/sr_imu)
    data_imu = f_imu(time_imu)

    f_ppg = scipy.interpolate.interp1d(time_ppg - time_ppg[0], data_ppg, axis=0)
    time_ppg = np.arange(0, time_ppg[-1] - time_ppg[0], 1/sr_ppg)
    data_ppg = f_ppg(time_ppg)
    return data_imu, data_ppg
def IMU_resample(data_imu, ):
    data_imu, time_imu = data_imu[:, 0], data_imu[:, -1]
    real_sr_imu = time_imu.shape[0]/ (time_imu[-1] - time_imu[0]) 

    data_imu, time_imu = revise_timestampe(data_imu, time_imu)
    f_imu = scipy.interpolate.interp1d(time_imu - time_imu[0], data_imu, axis=0)
    time_imu = np.arange(0, time_imu[-1] - time_imu[0], 1/sr_imu)
    data_imu = f_imu(time_imu)
   
    return data_imu
def mic_filter(data_mic, heartbeat_imu):
    # filter1: use IMU as reference
    # heartbeat_mic = [int(idx * sr_mic / sr_imu) for idx in heartbeat_imu]
    # gain = np.zeros_like(data_mic) + 0.2
    # for i in range(len(heartbeat_mic)):
    #     gain[heartbeat_mic[i] - 6000:heartbeat_mic[i] + 6000] = 1
    # data_mic = data_mic * gain
    # filter2: Wavelet Denoising
    data_mic = wtdenoise.get_baseline(data_mic)
    # data_mic = wtdenoise.tsd(data_mic)

    return data_mic 
def process_experiment(data_imu, data_mic, data_ppg):
    heartbeat_imu = heartbeat_segment.heart_rate_estimation(data_imu, plot=False)
    heartbeat_ppg = ppg_pipeline.pipeline(data_ppg)
    data_mic = mic_filter(data_mic, heartbeat_imu)

    return heartbeat_imu, data_mic, heartbeat_ppg
def process_playback(data_imu, data_mic,):
    heartbeat_imu = heartbeat_segment.heart_rate_estimation(data_imu, plot=False)
    data_mic = mic_filter(data_mic, heartbeat_imu)

    return heartbeat_imu, data_mic