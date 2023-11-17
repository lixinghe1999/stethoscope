import librosa
import os
import sys
sys.path.append('../')
from utils import *
import matplotlib.pyplot as plt
from deep_learning.metrics import AudioMetrics
playback_dir = 'measurement/playback/'
recording_dir = 'measurement/'
playback_file = 'heartbeat'

reference, sr = librosa.load(playback_dir + playback_file + '.wav', sr=None)
recordings = os.listdir(recording_dir)
recordings = [r for r in recordings if r.startswith(playback_file) and r.endswith('.wav')]
metric = AudioMetrics(sr)
fig, axs = plt.subplots(len(recordings), 1, figsize=(3, 6))

for i, recording in enumerate(recordings):

    _, device, medium, volume, number = recording[:-4].split('_')
    recording, _ = librosa.load(recording_dir + recording, sr=sr)
    if len(recording) > len(reference):
        correlation = np.correlate(recording, reference, mode='valid')
        shift = np.argmax(correlation)
        right_pad = len(recording) - shift - len(reference)
        recording = recording[shift: -right_pad]
    cos_sim = np.dot(recording, reference) / (np.linalg.norm(recording) * np.linalg.norm(reference))
    error = metric.evaluation(recording, reference)
    print('cos_sim:', cos_sim, 'error:', error)
    axs[i].plot(reference, label='reference', c='r')
    axs[i].plot(recording, label='recording', c='b')
    axs[i].set_title(device + ' ' + medium + ' ' + volume)
plt.legend()
plt.show()
