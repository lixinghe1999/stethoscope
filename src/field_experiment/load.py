'''
1. synchronize and resample
2. get the transfer function
'''
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy
import soundfile as sf
import sys
from tqdm import tqdm
sys.path.append('../')
from utils import *

import warnings
warnings.filterwarnings("ignore")
filter_list = {'heartbeat pure': scipy.signal.butter(4, 400, 'lowpass', fs=sr_mic),
               'heartbeat filtered': scipy.signal.butter(4, [30, 500], 'bandpass', fs=sr_mic),
               'imu': scipy.signal.butter(4, [1, 100], 'bandpass', fs=sr_imu)}
sr_mic = 4000
sr_imu = 400
sr_ppg = 25
def transfer_function(record, reference, ax):

    def spectrum(data1, data2):
        ps1 = np.fft.fft(data1)
        ps2 = np.fft.fft(data2)
        freq = np.fft.fftfreq(len(data1), 1/sr_mic)
        return freq, ps1, ps2
    freq, ps1, ps2 = spectrum(record, reference)
    ps1 = np.convolve(ps1, np.ones(100)/100, mode='same')
    ps2 = np.convolve(ps2, np.ones(100)/100, mode='same')
    res = ps1 / ps2
    max_freq = int(500 / (sr_mic / len(record)))
    res[max_freq:] = 0
    res_time = np.fft.ifft(res)
    reconstruciton = np.convolve(reference, res_time, mode='full')[:len(reference)]
    reconstruciton *= np.max(np.abs(record)) / np.max(np.abs(reconstruciton))
    cos_sim = abs(np.dot(reconstruciton, record) / (np.linalg.norm(reconstruciton) * np.linalg.norm(record)))
    print('response-based reconstruction', cos_sim)
    ax[0].plot(freq[:len(record)//2], np.abs(ps1)[:len(record)//2], label='record')
    ax[0].plot(freq[:len(record)//2], np.abs(ps2)[:len(record)//2], label='reference')
    ax[1].plot(record)
    ax[1].plot(reconstruciton)
    ax[1].set_ylim([-1, 1])
    return freq, res
def visual_imu(data_mic, axs):
    '''
    two axes to show microphone data in waveform and STFT, apply to smartphone or stethoscope recordings
    '''
    t_mic = np.arange(len(data_mic)) / sr_imu
    axs.plot(t_mic, data_mic)
    axs.set_ylim([-1, 1])
def visual_mic(data_mic, axs):
    '''
    two axes to show microphone data in waveform and STFT, apply to smartphone or stethoscope recordings
    '''
    t_mic = np.arange(len(data_mic)) / sr_mic
    f, t, mic_stft = scipy.signal.stft(data_mic, fs=sr_mic, nperseg=2048,)
    fmax = 800
    axs[0].plot(t_mic, data_mic)
    axs[0].set_ylim([-1, 1])
    axs[1].pcolormesh(t, f, np.abs(mic_stft))
    axs[1].set_ylim([0, fmax])

def visual_mic_ref(data_mic1, data_mic2, axs):
    scale = np.max(np.abs(data_mic1)) / np.max(np.abs(data_mic2))
    data_mic2 = data_mic2 * scale
    t_mic = np.arange(len(data_mic1)) / sr_mic
    
    axs.plot(t_mic, data_mic1)
    axs.plot(t_mic, data_mic2)

def load_data(sub_dir, save_sub_dir, save=False):
    if not os.path.exists(sub_dir):
        return []
    os.makedirs(save_sub_dir, exist_ok=True)
    files = os.listdir(sub_dir)
    files_mic = [f for f in files if f.split('_')[0] == 'MIC']
    references = [f for f in files if f.split('_')[0] == 'Steth']
    files_imu = [] # [f for f in files if f.split('_')[0] == 'IMU']
    number_of_files = len(files_mic)
    length = []
    metrics = []
    for i in range(0, number_of_files):
        mic = os.path.join(sub_dir, files_mic[i])
        reference = os.path.join(sub_dir, references[i])

        data_mic, sr = librosa.load(mic, sr=sr_mic)
        data_reference, sr = librosa.load(reference, sr=sr_mic)
        data_mic = scipy.signal.filtfilt(*filter_list['heartbeat filtered'], data_mic)
        data_reference = scipy.signal.filtfilt(*filter_list['heartbeat filtered'], data_reference)

        if len(files_imu) > i:
            data_imu = np.loadtxt(os.path.join(sub_dir, files_imu[i]), skiprows=1, delimiter=',', converters={1: converter})
            data_imu = IMU_resample(data_imu)
            data_imu = scipy.signal.filtfilt(*filter_list['imu'], data_imu)
        else: 
            data_imu = None    
        data_mic, data_reference, data_imu, metric = synchronize_playback(data_mic, data_reference, data_imu)
        metrics.append(metric)
        if save:
            data = np.vstack((data_mic, data_reference))
            fname = os.path.join(save_sub_dir, files_mic[i].replace('wav', 'flac').replace('MIC', 'Stereo'))
            sf.write(fname, data.T, sr_mic)
            length.append(len(data_mic)/sr_mic)
        else:
            fig, axs = plt.subplots(3, 2)
            visual_mic(data_mic, axs[0])
            visual_mic(data_reference, axs[1])
            visual_mic_ref(data_mic, data_reference, axs[2, 0])
            if len(files_imu) > i:
                visual_imu(data_imu, axs[2, 1])
            # transfer_function(data_mic, data_reference, axs[3])
            plt.show()
        # break
    if save:
        np.savetxt(os.path.join(save_sub_dir, 'reference.txt'), length, fmt='%1.2f')
    return metrics
        
if __name__ == "__main__":
    people = [
        # 'Lixing_He',
         # 'Liangyu_Liu'
         'Xuefu_Dong']
    # smartphone = ['iPhone13']
    smartphone = ['Pixel6', 'iPhone13', 'PixelXL']
    textile = ['skin', 'cotton', 'polyester', 'thickcotton', 'thickpolyester']
    for p in people:
        for s in smartphone:
            for t in textile:
                sub_dir = '_'.join([s, t, 'vertical'])
                metrics = load_data(os.path.join('thinklabs', p, sub_dir), 
                                    os.path.join('thinklabs_processed', p, sub_dir), save=True)
                if len(metrics) > 0:
                    print(sub_dir, np.mean(metrics, axis=0))