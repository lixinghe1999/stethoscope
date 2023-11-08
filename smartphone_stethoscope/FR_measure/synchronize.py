import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

def get_groundtruth(record, playback):
    correlation = np.correlate(record, playback, mode='valid')
    shift = np.argmax(correlation)
    right_pad = len(record) - shift - len(playback)
    return np.pad(playback, (shift, right_pad))
heartbeat, sr = sf.read('heartbeat_playback.wav')
heartbeat_upsample = np.interp(np.linspace(0, len(heartbeat), int(len(heartbeat) * 44100/4000)), np.arange(len(heartbeat)), heartbeat)
heartbeat_record, sr = sf.read('data\heartbeat_smartphone_contact.wav')
gt = get_groundtruth(heartbeat_record, heartbeat_upsample)

plt.plot(heartbeat_record)
# plt.plot(heartbeat_upsample)
plt.plot(gt)
plt.show()