'''
according to "iStethoscope: A Demonstration of the Use of Mobile Devices for Auscultation"
'''
import numpy as np
import scipy
sr_mic = 48000
sr_imu = 400
filter_list = {'heartbeat pure': scipy.signal.butter(4, 400, 'lowpass', fs=sr_mic),
               'heartbeat filtered': scipy.signal.butter(4, [20, 1000], 'bandpass', fs=sr_mic),
               'imu': scipy.signal.butter(4, [1, 25], 'bandpass', fs=sr_imu)}



