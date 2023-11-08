import sounddevice as sd
import soundfile as sf
import scipy
import time
import matplotlib.pyplot as plt

# set volume to 2%
# chirp, sr = sf.read('chirp_playback.wav')
# duration = len(chirp) // sr
# # sd.default.device = 'Headphones (BTS0011)'
# myrecording = sd.playrec(chirp, sr, channels=1, blocking=True)
# scipy.io.wavfile.write('chirp_stethoscope.wav', sr, myrecording)  
# sd.stop()

# set volume to 15%
heartbeat, sr = sf.read('heartbeat_playback.wav')
duration = len(heartbeat) // sr
# sd.default.device = 'Headphones (BTS0011)'
myrecording = sd.playrec(heartbeat, sr, channels=1, blocking=True)
scipy.io.wavfile.write('heartbeat_stethoscope.wav', sr, myrecording)  
