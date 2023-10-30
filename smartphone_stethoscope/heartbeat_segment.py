'''
we separate heartbeat cycles
'''
import scipy
import numpy as np
import matplotlib.pyplot as plt

def heart_rate_estimation(sequence, sr=400, plot=False):
    peaks, property = scipy.signal.find_peaks(sequence, prominence=0.05, distance=10, width=1)
    sort_peaks = peaks[np.argsort(sequence[peaks])[::-1]]
    candidate_set = []
    for i in range(len(sort_peaks)):
        candidate_set.append(sort_peaks[i])
        for j in candidate_set[:-1]:
            if abs(sort_peaks[i] - j) < (sr * 0.3):
                candidate_set.pop()
                break
    candidate_set = np.array(candidate_set)
    candidate_set = np.sort(candidate_set)
    close_index = np.argmin(np.abs(np.diff(candidate_set)))
    interval = candidate_set[close_index + 1] - candidate_set[close_index]
    coarse_template = [candidate_set[close_index] - interval//2,  candidate_set[close_index + 1] + interval//2]

    correlation = np.correlate(sequence, sequence[coarse_template[0]:coarse_template[1]], mode='same')
    correlation = correlation / np.max(correlation)
    peaks_correlation, property = scipy.signal.find_peaks(correlation, height=0.5, distance=10, width=1)
    segmentation = np.stack([peaks_correlation - interval//2, peaks_correlation + interval//2])
    if plot:
        fig, axs = plt.subplots(4, 1)
        axs[0].plot(sequence)
        axs[0].plot(peaks, sequence[peaks], "x")
        axs[0].plot(candidate_set, sequence[candidate_set], "o")
        axs[0].plot(coarse_template, sequence[coarse_template], "*")
        axs[1].plot(sequence[coarse_template[0]:coarse_template[1]])
        axs[2].plot(correlation)
        axs[2].plot(peaks_correlation, correlation[peaks_correlation], "x")
        for i in range(len(peaks_correlation)):
            axs[3].plot(sequence[segmentation[0, i]:segmentation[1, i]])
    HRV = 60 / (np.mean(np.diff(peaks_correlation)) / sr)
    print("IMU:", HRV)
    return peaks_correlation