'''
1. Prepare chirp and heartbeat sample for measurment
2. sampler function to access the data from public dataset
'''
import scipy
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
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
def parse_CHSC(parent_dir, choose_set='set_a'):
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
def parse_Cardiology_Challenge_2016(parent_dir, choose_set='training-a'):
    '''
    Classification of Heart Sound Recordings: The PhysioNet/Computing in Cardiology Challenge 2016
    DATASET website: https://physionet.org/content/challenge-2016/1.0.0/
    '''
    loc = os.path.join(parent_dir, 'Cardiology_Challenge_2016', choose_set)
    reference = np.loadtxt(os.path.join(loc, 'REFERENCE.csv'), delimiter=',', dtype=str)
    audio_files = []
    labels = []
    for ref in reference:
        f, label = ref
        audio_files.append(os.path.join(loc, f+'.wav'))
        labels.append(int(label))
    return audio_files, labels
def parse_PhysioNet(parent_dir, choose_set='training-a'):
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

def parse_CirCor(parent_dir, choose_set='training_data'):
    '''
    The CirCor DigiScope Phonocardiogram Dataset from PhysioNet
    https://physionet.org/content/circor-heart-sound/1.0.3/
    '''
    loc = os.path.join(parent_dir, 'CirCor', choose_set)
    audio_files = os.listdir(loc)
    audio_files = [os.path.join(loc, f) for f in audio_files if f.split('.')[-1] == 'wav']
    reference = np.loadtxt(loc + '.csv', delimiter=',', dtype=str)
    labels = []
    return audio_files, labels, 

def parse_ephongram(parent_dir, choose_set='WAV'):
    '''
    https://physionet.org/content/ephnogram/1.0.0/
    '''
    loc = os.path.join(parent_dir, 'ephnogram', choose_set)
    audio_files = os.listdir(loc)
    audio_files = [os.path.join(loc, f) for f in audio_files if f.split('.')[-1] == 'wav']
    labels = []
    return audio_files, labels

def parse_radarheart(parent_dir, choose_set='WAV'):
    '''
    https://www.nature.com/articles/s41597-020-0390-1
    '''
    loc = os.path.join(parent_dir, 'radarheart', choose_set)
    audio_files = os.listdir(loc)
    audio_files = [os.path.join(loc, f) for f in audio_files if f.split('.')[-1] == 'wav']
    labels = []
    return audio_files, labels

def parse_Respiratory(parent_dir, choose_set='audio_and_txt_files'):
    '''
    https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database/data
    '''
    loc = os.path.join(parent_dir, 'Respiratory', choose_set)
    audio_files = os.listdir(loc)
    audio_files = [os.path.join(loc, f) for f in audio_files if f.split('.')[-1] == 'wav']
    labels = []
    return audio_files, labels
def parse_lungsound(parent_dir, choose_set='Audio Files'):
    '''
    https://data.mendeley.com/datasets/jwyy9np4gv/3
    '''
    loc = os.path.join(parent_dir, 'lungsound', choose_set)
    audio_files = os.listdir(loc)
    audio_files = [os.path.join(loc, f) for f in audio_files if f.split('.')[-1] == 'wav']
    labels = []
    return audio_files, labels
target_sr = 4000
import librosa
import soundfile as sf
def resample(audio_files, folder):
    os.makedirs(folder, exist_ok=True)
    length = []
    for f in tqdm(audio_files):
        data, sr = librosa.load(f, sr=target_sr)
        length.append(len(data)/sr)
        f_new = os.path.basename(f).split('.')[0] + '.flac'
        f_new = os.path.join(folder, f_new)
        sf.write(f_new, data, sr)
    np.savetxt(os.path.join(folder, 'reference.txt'), length, fmt='%.4f')
if __name__ == "__main__":
    compress = True
    if compress:
        ori_folder = 'public_dataset'
        target_folder = 'normalized_dataset'

        # audio_files, labels = parse_Respiratory(ori_folder)
        # resample(audio_files, os.path.join(target_folder, 'Respiratory', 'audio_and_txt_files'))

        # audio_files, labels = parse_lungsound(ori_folder)
        # resample(audio_files, os.path.join(target_folder, 'lungsound', 'Audio Files'))

        # for s in ['set_a', 'set_b']:
        #     audio_files, labels = parse_CHSC(ori_folder, choose_set=s)
        #     print('CHSC', s, len(audio_files))
        #     resample(audio_files, os.path.join(target_folder, 'CHSC', s))

        # for s in ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f', 'validation']:
        #     audio_files, labels = parse_PhysioNet('public_dataset', choose_set=s)
        #     print('PhysioNet', s, len(audio_files)) 
        #     resample(audio_files, os.path.join(target_folder, 'PhysioNet', s))
        
        # audio_files, labels = parse_Thinklabs('public_dataset')
        # print('Thinklabs', len(audio_files))
        # resample(audio_files, os.path.join(target_folder, 'Thinklabs', 'wav'))

        # audio_files, labels = parse_CirCor('public_dataset')
        # print('CirCor', len(audio_files))
        # resample(audio_files, os.path.join(target_folder, 'CirCor', 'training_data'))

        # audio_files, labels = parse_ephongram('public_dataset')
        # print('ephongram', len(audio_files))
        # resample(audio_files, os.path.join(target_folder, 'ephongram', 'WAV'))

        audio_files, labels = parse_radarheart('public_dataset')
        print('radarheart', len(audio_files))
        resample(audio_files, os.path.join(target_folder, 'radarheart', 'WAV'))
    else:
        ori_folder = 'public_dataset'
        target_folder = 'smartphone'
        for s in ['set_a', 'set_b']:
            audio_files, labels = parse_CHSC(ori_folder, choose_set=s)
            print('CHSC', s, len(audio_files))
            os.makedirs(os.path.join(target_folder, 'CHSC'), exist_ok=True)
            os.makedirs(os.path.join(target_folder, 'CHSC', s), exist_ok=True)
            np.savetxt(os.path.join(target_folder, 'CHSC', s, 'reference.txt'), audio_files, fmt='%d')
            
        # for s in ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f', 'validation']:
        #     audio_files, labels = parse_PhysioNet('public_dataset', choose_set=s)
        #     print('PhysioNet', s, len(audio_files)) 
        #     os.makedirs(os.path.join(target_folder, 'PhysioNet'), exist_ok=True)
        #     os.makedirs(os.path.join(target_folder, 'PhysioNet', s), exist_ok=True)
        #     np.savetxt(os.path.join(target_folder, 'PhysioNet', s, 'reference.txt'), audio_files, fmt='%d')
        
        # audio_files, labels = parse_Thinklabs('public_dataset')
        # print('Thinklabs', len(audio_files))
        # os.makedirs(os.path.join(target_folder, 'Thinklabs'), exist_ok=True)
        # os.makedirs(os.path.join(target_folder, 'Thinklabs', 'wav'), exist_ok=True)
        # np.savetxt(os.path.join(target_folder, 'Thinklabs', 'wav', 'reference.txt'), audio_files, fmt='%d')

        # audio_files, labels = parse_CirCor('public_dataset')
        # print('CirCor', len(audio_files))
        # os.makedirs(os.path.join(target_folder, 'CirCor'), exist_ok=True)
        # os.makedirs(os.path.join(target_folder, 'CirCor', 'training_data'), exist_ok=True)
        # np.savetxt(os.path.join(target_folder, 'CirCor', 'training_data', 'reference.txt'), audio_files, fmt='%d')

        # audio_files, labels = parse_ephongram('public_dataset')
        # print('ephongram', len(audio_files))
        # os.makedirs(os.path.join(target_folder, 'ephongram'), exist_ok=True)
        # os.makedirs(os.path.join(target_folder, 'ephongram', 'WAV'), exist_ok=True)
        # np.savetxt(os.path.join(target_folder, 'ephongram', 'WAV', 'reference.txt'), audio_files, fmt='%d')
    


  
    