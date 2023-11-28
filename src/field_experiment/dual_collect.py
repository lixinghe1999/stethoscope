'''
1. adb connect
2. record audio on PC
'''
import sounddevice as sd
from scipy.io.wavfile import write
import datetime
import android_controller
import time

time.sleep(1) # wait for 1 second to prepare
fs = 44100  # Sample rate
seconds = 10  # Duration of recording

devices = android_controller.checkConnections()
'''
turn on Mobile Hotspot and connect
adb tcpip 5555
adb connect IP Address
'''
# 750 750 Pixel XL
# 550 650 Huawei MATE20, Pixel 6
android_controller.tap(750, 750)
now_time = datetime.datetime.now()
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, blocking=True)
android_controller.tap(750, 750)

str_time = datetime.datetime.strftime(now_time, '%Y%m%d_%H%M%S_%f')
filename = 'Steth_' + str_time + '.wav'
write(filename, fs, myrecording)  # Save as WAV file
