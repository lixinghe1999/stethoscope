'''
1. adb connect
2. record audio on PC
'''
import sounddevice as sd
import scipy
import numpy as np
import soundfile as sf
import android_controller
import datetime
def measure_smartphone(wav='chirp_playback.wav', sr=None):
    if type(wav) == str:
        recording, sr = sf.read(wav) # else, directly use numpy array
    else:
        assert sr is not None
    b, a = scipy.signal.butter(4, [25, 800], 'bandpass', fs=sr)
    recording = scipy.signal.filtfilt(b, a, recording)
    recording = (recording - np.mean(recording))/np.max(np.abs(recording))
    # android_controller.connect('192.168.137.131:5555') # the IP address can change. either wireless or wired connection
    # 750 750 Pixel XL
    # 550 650 Huawei MATE20, Pixel 6
    android_controller.tap(750, 750)
    sd.play(recording, sr, blocking=True)
    android_controller.tap(750, 750)
def measure_stethoscope(wav='chirp_playback.wav', sr=None):
    if type(wav) == str:
        recording, sr = sf.read(wav) # else, directly use numpy array
    else:
        assert sr is not None
    b, a = scipy.signal.butter(4, [25, 800], 'bandpass', fs=sr)
    recording = scipy.signal.filtfilt(b, a, recording)
    recording = (recording - np.mean(recording))/np.max(np.abs(recording))
    now_time = datetime.datetime.now()
    str_time = datetime.datetime.strftime(now_time, '%Y%m%d_%H%M%S_%f')
    filename = 'Steth_' + str_time + '.wav'
    myrecording = sd.playrec(recording, sr, channels=1, blocking=True)
    scipy.io.wavfile.write(filename, sr, myrecording)
    return myrecording
if __name__ == "__main__":
    directory = 'measurement/playback/'
    play_file = 'Steth1.wav'
    # measure_smartphone(directory + play_file) # 2% volume
    measure_stethoscope(directory + play_file)

