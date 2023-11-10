'''
1. adb connect
2. record audio on PC
'''
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
import android_controller
def measure_smartphone(wav='chirp_playback.wav', sr=None):
    if type(wav) == str:
        recording, sr = sf.read(wav) # else, directly use numpy array
    else:
        assert sr is not None
    # android_controller.connect('192.168.137.131:5555') # the IP address can change. either wireless or wired connection
    devices = android_controller.checkConnections()
    android_controller.tap(750, 750)
    sd.play(recording, sr, blocking=True)
    android_controller.tap(750, 750)
def measure_stethoscope(wav='chirp_playback.wav', sr=None):
    if type(wav) == str:
        recording, sr = sf.read(wav) # else, directly use numpy array
    else:
        assert sr is not None
    myrecording = sd.playrec(recording, sr, channels=1, blocking=True)
    return myrecording
if __name__ == "__main__":
    # measure_smartphone('chirp_playback.wav') # 2% volume
    measure_smartphone('heartbeat_playback.wav') # 15% volume
    # measure_stethoscope('heartbeat_playback.wav')

