'''
1. adb connect
2. record audio on PC
'''
import sounddevice as sd
from scipy.io.wavfile import write
import datetime
import time
import os
fs = 44100  # Sample rate
seconds = 20  # Duration of recording
def dual_record(parent_dir='./'):
    os.system('adb devices')
    device_2 = sd.query_devices(2)
    assert device_2['name'] == 'Microphone (2- High Definition '
    print('good, confirmed!') # device_1 = Mic, device_2 = Stethoscope
    time.sleep(2)

    '''
    turn on Mobile Hotspot and connect
    adb tcpip 5555
    adb connect IP Address -> manually check
    '''
    save_dir = parent_dir.split('\\')[-1]
    # os.system("adb shell input keyevent --longpress 67 67 67 67 67 67 67 67 67 67 67 67 67 67 67 67 67 67")
    # os.system('adb shell input text ' + '"' + save_dir + '"')
    
    # '750 750' Pixel XL '
    # '550 650' Huawei MATE20, Pixel 6
    loc = '550 650'
    os.system('adb shell input tap ' + loc)
    now_time = datetime.datetime.now()
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, blocking=True, device=2)
    os.system('adb shell input tap ' + loc)

    str_time = datetime.datetime.strftime(now_time, '%H%M%S_%f')
    filename = os.path.join(parent_dir, 'Steth_' + str_time + '.wav')
    write(filename, fs, myrecording)  # Save as WAV file
if __name__ == '__main__':
    import sys
    args = sys.argv
    if len(args) == 1:
        dual_record()
    else:
        dual_record(args[1])