import numpy as np
import scipy
def moving_average(a, n=3):
    average = np.convolve(a, np.ones((n,))/n, mode='same')
    return a - average
def centeredifference(a):
    shift_a = np.roll(a, -2)
    a = (shift_a - a) / 2
    a[-2:] = 0
    return a
def MaximaCalculator(a):
    a = centeredifference(a)
    peaks = []
    for i in range(len(a)-1):
        if (a[i+1] * a[i]) < 0 and (a[i+1] - a[i]) < 0:
            peaks.append(i)
    return peaks
def pipeline(ppg, sr=25):
    ppg_raw = ppg
    ppg = moving_average(ppg_raw, 10)

    b, a = scipy.signal.butter(1, 4, 'lowpass', fs=25)
    ppg = scipy.signal.filtfilt(b, a, ppg)

    ppg = scipy.ndimage.gaussian_filter(ppg, 5)

    # ppg = ppg.reshape(chunk_size, -1)[:, 1:-1]
    # peaks = scipy.signal.find_peaks(ppg, height=0.1)[0]
    peaks = MaximaCalculator(ppg)
    HRV = 60 / (np.mean(np.diff(peaks)) / sr)
    print("PPG:", HRV)
    return ppg_raw, ppg, peaks
