'''
1. convert "EPHNOGRAM: A Simultaneous Electrocardiogram and Phonocardiogram Database" from MAT to audio
2. split other dataset into 10-seconds or less audio clips
'''
import scipy
import os
from tqdm import tqdm
length = 10
def convert_mat_to_wav(file, output):
    data = scipy.io.loadmat(file)
    fs = data['fs'][0][0]
    pcg = (data['PCG'][0] * 2**15).astype('int16')
    for i in range(0, len(pcg), length * fs):
        start = i
        end = min(i + length * fs, len(pcg))
        scipy.io.wavfile.write(output + '_' + str(i//(length*fs)) + '.wav', fs, pcg[start:end])
def convert_ephnogram():
    directory = 'public_dataset/ephnogram/MAT'
    target_directory = 'public_dataset/ephnogram/WAV'
    files = os.listdir(directory)
    for file in tqdm(files):
        convert_mat_to_wav(os.path.join(directory, file), os.path.join(target_directory, file[:-4])) 
def convert_radarheart():
    directory = 'public_dataset/radarheart'
    target_directory = 'public_dataset/radarheart/WAV'
    f = os.walk(directory)
    for path, dir_list, file_list in tqdm(f):
        for file in file_list:
            if file.endswith('.mat'):
                data = scipy.io.loadmat(path + '/' + file)
                fs = data['Fs'][0][0]
                pcg = (data['pcg_audio'] * 2**15).astype('int16')
                print(fs, pcg.shape)
                for i in range(0, len(pcg), length * fs):
                    start = i
                    end = min(i + length * fs, len(pcg))
                    scipy.io.wavfile.write(target_directory + '/' + file[:-4] + '_' + str(i//(length*fs)) + '.wav', fs, pcg[start:end])

    return 
if __name__ == "__main__":
    # convert_ephnogram()
    convert_radarheart()