'''
1. adb connect
2. record audio on PC
'''
import sounddevice as sd
import librosa
import scipy
import datetime
from multiprocessing import Process
seconds = 10  # Duration of recording
def record(f_name, y, fs):
    print(datetime.datetime.now())
    myrecording = sd.rec(len(y), samplerate=fs, channels=1, blocking=True)
    scipy.io.wavfile.write(f_name, fs, myrecording)
def play(y, fs):
    print(datetime.datetime.now())

    sd.play(y, fs, blocking=True)
    
def dual_record(f, f_name):
    y, fs = librosa.load(f, sr=16000)
    if len(y) > seconds*fs:
        y = y[:seconds*fs] # max 20 seconds
    print('good, confirmed!') # device_1 = Mic, device_2 = Stethoscope

    p1 = Process(target=record, args=[f_name, y, fs])
    p2 = Process(target=play, args=[y, fs])
    p1.start()
    p2.start()
    p1.join()
    p2.join()
if __name__ == '__main__':
    import sys
    args = sys.argv
    if len(args) == 1:
        dual_record('./dataset/chirp.wav', './dataset/chirp_Thinklabs_nothing.wav')
    else:
        dual_record(args[1], args[2])