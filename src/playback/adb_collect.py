'''
1. adb connect
2. record audio on PC
'''
import sounddevice as sd
import librosa
import scipy
import numpy as np
import time
import os
fs = 44100  # Sample rate
seconds = 20  # Duration of recording
def dual_record(f):
    os.system('adb devices')
    y, fs = librosa.load(f, sr=None)
    b, a = scipy.signal.butter(4, [25, 800], 'bandpass', fs=fs)
    y = scipy.signal.filtfilt(b, a, y)
    y = (y - np.mean(y))/np.max(np.abs(y))
    if len(y) > seconds*fs:
        y = y[:seconds*fs] # max 20 seconds
    print('good, confirmed!')
    '''
    turn on Mobile Hotspot and connect
    adb tcpip 5555
    adb connect IP Address -> manually check
    '''    
    # '750 750' Pixel XL '
    # '550 650' Huawei MATE20, Pixel 6
    loc = '750 750'
    os.system('adb shell input tap ' + loc)
    sd.play(y, fs, blocking=True)
    os.system('adb shell input tap ' + loc)

if __name__ == '__main__':
    import sys
    args = sys.argv
    if len(args) == 1:
        dual_record('./dataset/chirp.wav')
    else:
        dual_record(args[1])