import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from igor import Igor


def detect_peak(x, y, x_range, rocking: str = "006", order: str = "0"):
    """
    Detect peaks in a given x, y data set.
    """
    ideal_006_order = {
        "-2": 14.608,
        "-1": 15.744,
        "0": 17.386,
        "1": 19.052,
        "2": 20.192,
    }
    ideal_0015_order = {
        "-2": 41.5,
        "-1": 42.788,
        "0": 44.53,
        "1": 46.336,
        "2": 47.624,
    }

    # Get the indices of the x values within the range
    x_indices = np.where((x >= x_range[0]) & (x <= x_range[1]))[0]
    x = np.array(x)[x_indices]
    y = np.array(y)[x_indices]

    # Find the peaks
    # peaks = np.array(signal.find_peaks_cwt(y, np.arange(1, 100)))
    peaks = np.array(signal.find_peaks(y, distance=1))[0]

    # Get the peak x and y values
    peaks_x = x[peaks]
    peaks_y = y[peaks]

    if rocking == "006":
        # Find the closest peak to the ideal order 0
        closest_peak = np.argmin(np.abs(peaks_x - ideal_006_order[order]))
        peak_x = peaks_x[closest_peak]
        peak_y = peaks_y[closest_peak]
    elif rocking == "0015":
        # Find the closest peak to the ideal order 0
        closest_peak = np.argmin(np.abs(peaks_x - ideal_0015_order[order]))
        peak_x = peaks_x[closest_peak]
        peak_y = peaks_y[closest_peak]

    return peak_x, peak_y
