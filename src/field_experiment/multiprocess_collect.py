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
seconds = 20  # Duration of recording
def record(device=1, parent_dir='./', prefix='MIC_', channels=1):
    now_time = datetime.datetime.now()
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=channels, blocking=True, device=device)
    str_time = datetime.datetime.strftime(now_time, '%H%M%S_%f')
    f_name = os.path.join(parent_dir, prefix + str_time + '.wav')
    write(f_name, fs, myrecording)

def dual_record(parent_dir='./', phone='iPhone13'):
    devices = sd.query_devices()
    if phone in ['PixelXL', 'Pixel6', 'iPhone13']:
        device_name = 'Microphone (WO Mic Device)'
        device_channel = 1
    else:
        device_name = 'Desktop Microphone (RØDE AI-Mic'
        device_channel = 2
    for d in devices:
        if d['hostapi'] == 0:
            # first, search the stethoscope
            if 'Microphone (2- High Definition ' == d['name']:
                steth_channel = 1
                steth_index = d['index']
                print('Stethoscope', d['index'])
            elif device_name == d['name']:
                device_index = d['index']
                print('Device', device_name, d['index'])
            
    # device_1 = sd.query_devices(1)
    # device_2 = sd.query_devices(2)
    # print(device_1['name'], device_2['name'])
    # if phone in ['PixelXL', 'Pixel6', 'iPhone13']:   
    #     assert device_1['name'] == 'Microphone (WO Mic Device)'
    #     assert device_2['name'] == 'Microphone (2- High Definition '
    # else:
    #     assert device_1['name'] == 'Microphone (WO Mic Device)'
    #     assert device_2['name'] == 'Microphone (2- High Definition '
    print('good, confirmed!')

    time.sleep(2)
    p1 = Process(target=record, args=[device_index, parent_dir, 'MIC_', device_channel])
    p2 = Process(target=record, args=[steth_index, parent_dir, 'Steth_', steth_channel])
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
        dual_record(args[1], args[2])
