'''
1. Prepare chirp and heartbeat sample for measurment
2. sampler function to access the data from public dataset
'''
import scipy
import numpy as np
import matplotlib.pyplot as plt
import os
def plot_spectrogram(title, w, fs, ax):
    ff, tt, Sxx = scipy.signal.spectrogram(w, fs=fs, nperseg=256, nfft=512)
    ax.pcolormesh(tt, ff, Sxx, cmap='gray_r', shading='gouraud')
    ax.set_title(title)
    ax.set_xlabel('t (sec)')
    ax.set_ylabel('Frequency (Hz)')
    ax.grid(True)
def prepare_chirp():    
    fig, ax = plt.subplots()
    fs = 44100
    T = 10
    t = np.arange(0, int(T*fs)) / fs
    chirp = scipy.signal.chirp(t, f0=25, f1=800, t1=T, method='linear')

    plot_spectrogram('Linear Chirp', chirp, fs, ax)
    scipy.io.wavfile.write('chirp_playback.wav', fs, chirp)
    plt.show()
def prepare_heartbeat():
    fig, ax = plt.subplots()

    fs, heartbeat = scipy.io.wavfile.read('heartbeat.wav')
    b, a = scipy.signal.butter(4, [25, 800], 'bandpass', fs=fs)
    heartbeat = scipy.signal.filtfilt(b, a, heartbeat)
    heartbeat = (heartbeat - np.mean(heartbeat))/np.max(np.abs(heartbeat))

    plot_spectrogram('Heartbeat', heartbeat, fs, ax)
    scipy.io.wavfile.write('heartbeat_playback.wav', fs, heartbeat)
    plt.show()
def parser_CHSC(parent_dir, choose_set='set_a'):
    '''
    DATASET website: 
    1. https://istethoscope.peterjbentley.com/heartchallenge/index.html
    2. https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds/data
    Description:
    set_a: iStethoscope Pro app on iPhone
    set_b: digital stethoscope DigiScope
    '''
    loc = os.path.join(parent_dir, 'CHSC', choose_set)
    audio_files = os.listdir(loc)
    labels = []
    if choose_set == 'set_a':
        map_dict = {'Aunlabelledtest': 0, 'extrahls': 1, 'artifact': 2, 'murmur': 3, 'normal': 4}
    else:
        map_dict = {'Bunlabelledtest': 0, 'extrastole': 1, 'murmur': 2, 'normal': 3}
    for f in audio_files:
        label = map_dict[f.split('_')[0]]
        labels.append(label)
    audio_files = [os.path.join(loc, f) for f in audio_files]
    return audio_files, labels
def parser_PhysioNet(parent_dir, choose_set='training-a'):
    '''
    DATASET website: https://physionet.org/content/challenge-2016/1.0.0/
    '''
    loc = os.path.join(parent_dir, 'PhysioNet', choose_set)
    reference = np.loadtxt(os.path.join(loc, 'REFERENCE.csv'), delimiter=',', dtype=str)
    audio_files = []
    labels = []
    for ref in reference:
        f, label = ref
        audio_files.append(os.path.join(loc, f+'.wav'))
        labels.append(int(label))
    return audio_files, labels
def parser_PhysioNet(parent_dir, choose_set='training-a'):
    '''
    DATASET website: https://physionet.org/content/challenge-2016/1.0.0/
    '''
    loc = os.path.join(parent_dir, 'PhysioNet', choose_set)
    reference = np.loadtxt(os.path.join(loc, 'REFERENCE.csv'), delimiter=',', dtype=str)
    audio_files = []
    labels = []
    for ref in reference:
        f, label = ref
        audio_files.append(os.path.join(loc, f+'.wav'))
        labels.append(int(label))
    return audio_files, labels
def parse_Thinklabs(parent_dir, choose_set='wav'):
    '''
    Thinklabs One dataset from Youtube
    https://www.youtube.com/channel/UCzEbKuIze4AI1523_AWiK4w/videos
    Note that there is not original labels for this dataset
    '''
    loc = os.path.join(parent_dir, 'Thinklabs', choose_set)
    audio_files = os.listdir(loc)
    audio_files = [os.path.join(loc, f) for f in audio_files]
    labels = [0] * len(audio_files)
    return audio_files, labels

if __name__ == "__main__":
    # measurement_chirp()
    # measurement_heartbeat()

    audio_files, labels = parser_CHSC('../public_dataset', choose_set='set_b')
    print(len(audio_files))

    audio_files, labels = parser_PhysioNet('../public_dataset', choose_set='training-a')
    print(len(audio_files))

    audio_files, labels = parse_Thinklabs('../public_dataset')
    print(len(audio_files))

  
    