'''
Large-scale data collection
1. sample data from ../public_dataset
2. play and record it with "measure_smartphone" or "measure_stethoscope"
3.1. stethoscope, save playback and recording with timestamps (only for evaluation and measurement)
3.2. smartphone, save recording with timestamps, synchronize with playback offline, also record the corresponding playback
'''
from playback_prepare import parse_Thinklabs, parser_CHSC, parser_PhysioNet, parse_CirCor
import sounddevice as sd
import android_controller
import scipy
import numpy as np
import argparse
import os
import time
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='CHSC', required=True, choices=['CHSC', 'PhysioNet', 'Thinklabs'])
    parser.add_argument('-s', '--choose_set', type=str, default='set_a', required=True)
    args = parser.parse_args()

    source_dataset = 'public_dataset'
    if args.dataset == 'CHSC':
        assert args.choose_set in ['set_a', 'set_b']
        audio_files, labels, lengths = parser_CHSC(source_dataset, args.choose_set)
    elif args.dataset == 'PhysioNet':
        assert args.choose_set in ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f', 'validation']
        audio_files, labels, lengths = parser_PhysioNet(source_dataset, args.choose_set)
    elif args.dataset == 'Thinklabs':
        assert args.choose_set in ['wav']
        audio_files, labels, lengths = parse_Thinklabs(source_dataset, args.choose_set)

    print(args.dataset, args.choose_set, len(audio_files))

    target_dataset = 'smartphone'


    os.makedirs(target_dataset, exist_ok=True)
    os.makedirs(os.path.join(target_dataset, args.dataset), exist_ok=True)
    os.makedirs(os.path.join(target_dataset, args.dataset, args.choose_set), exist_ok=True)

    file = open(os.path.join(target_dataset, args.dataset, args.choose_set, 'reference.txt'), 'w')
    for audio, label, length in zip(audio_files, labels, lengths):
        file.write(audio + ' ' + str(label) + ' ' + str(length) +"\n")
    file.close()
    for i, f in enumerate(audio_files):
        print(i, f, len(audio_files))
        fs, heartbeat = scipy.io.wavfile.read(f)
        b, a = scipy.signal.butter(4, [25, 800], 'bandpass', fs=fs)
        heartbeat = scipy.signal.filtfilt(b, a, heartbeat)
        heartbeat = (heartbeat - np.mean(heartbeat))/np.max(np.abs(heartbeat))

        # the IP address can change. either wireless or wired connection
        # use adb connect XXXXX first
        # android_controller.connect('192.168.137.235:5555') 
        devices = android_controller.checkConnections()
        android_controller.tap(750, 750)
        sd.play(heartbeat, fs, blocking=True)
        android_controller.tap(750, 750)
        duration = len(heartbeat)/fs
        save_time = max(1, duration//2)
        time.sleep(save_time) # wait until finishing saving, may do something to accelerate it

        # break
