'''
1. adb connect
2. record audio on PC
'''
import sounddevice as sd
from scipy.io.wavfile import write
import datetime
import android_controller
import time

time.sleep(3) # wait for 1 second to prepare

now_time = datetime.datetime.now()
str_time = datetime.datetime.strftime(now_time, '%Y%m%d_%H%M%S_%f')
filename = 'SCIAN_' + str_time + '.wav'
fs = 44100  # Sample rate
seconds = 10  # Duration of recording

devices = android_controller.checkConnections()
print(devices)
# android_controller.connect('192.168.137.71:5555') 
# start or stop the android app
android_controller.tap(750, 750)
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, blocking=True)
android_controller.tap(750, 750)

write(filename, fs, myrecording)  # Save as WAV file
