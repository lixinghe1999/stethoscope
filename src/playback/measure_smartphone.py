'''
1. adb connect
2. record audio on PC
'''
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
import android_controller
import time


chirp, sr = sf.read('chirp_playback.wav') # 2%
#chirp, sr = sf.read('heartbeat_playback.wav') # 15% 

duration = len(chirp) // sr

android_controller.connect('192.168.137.131:5555') # the IP address can change. either wireless or wired connection
devices = android_controller.checkConnections()
print(devices)


android_controller.tap(750, 750)
sd.play(chirp, sr, blocking=True)
android_controller.tap(750, 750)



