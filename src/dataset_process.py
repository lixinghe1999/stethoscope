'''
convert "EPHNOGRAM: A Simultaneous Electrocardiogram and Phonocardiogram Database" from MAT to audio
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

if __name__ == "__main__":
    directory = 'public_dataset/ephnogram/MAT'
    target_directory = 'public_dataset/ephnogram/WAV'
    files = os.listdir(directory)
    for file in tqdm(files):
        convert_mat_to_wav(os.path.join(directory, file), os.path.join(target_directory, file[:-4])) 
