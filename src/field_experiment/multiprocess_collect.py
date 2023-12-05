'''
use WO MIC to turn Android and IOS smartphone into PC microphone

'''
import sounddevice as sd
import datetime
from scipy.io.wavfile import write
from multiprocess import Process
import time
import os
fs = 44100  # Sample rate
seconds = 10  # Duration of recording
def record(device=1, parent_dir='./'):
    now_time = datetime.datetime.now()
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, blocking=True, device=device)
    str_time = datetime.datetime.strftime(now_time, '%H%M%S_%f')
    if device == 1:
        f_name = os.path.join(parent_dir, 'MIC_' + str_time + '.wav')
        write(f_name, fs, myrecording)
    else:
        f_name = os.path.join(parent_dir, 'Steth_' + str_time + '.wav')
        write(f_name, fs, myrecording)

def dual_record(parent_dir='./'):
    device_1 = sd.query_devices(1)
    device_2 = sd.query_devices(2)
    assert device_1['name'] == 'Microphone (WO Mic Device)'
    assert device_2['name'] == 'Microphone (2- High Definition '
    print('good, confirmed!') # device_1 = Mic, device_2 = Stethoscope

    time.sleep(2)
    p1 = Process(target=record, args=[1, parent_dir])
    p2 = Process(target=record, args=[2, parent_dir])
    p1.start()
    p2.start()
    p1.join()
    p2.join()
if __name__ == '__main__':
    import sys
    args = sys.argv
    if len(args) == 1:
        dual_record()
    else:
        dual_record(args[1])
