import numpy as np
from scipy.signal import butter, lfilter

def butter_highpass_filter(data, highcut, dt, order=5):
    fs = 1. / dt
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='highpass')
    y = lfilter(b, a, data)
    return y


def ricker(freq, length, dt, peak_time):
    """Return a Ricker wavelet with the specified central frequency.
    
    Args:
        freq: A float specifying the central frequency of the wavelet
        length: An integer specifying the number of time steps to use
        dt: A float specifying the time interval between time steps
        peak_time: A float specifying the time (in time units) at which the
                   peak amplitude of the wavelet occurs

    Returns:
        A 1D Numpy array of length 'length' containing a Ricker wavelet
    """
    t = (np.arange(length) * dt - peak_time).astype(np.float32)
    y = (1 - 2 * np.pi**2 * freq**2 * t**2) \
            * np.exp(-np.pi**2 * freq**2 * t**2)
   # y = butter_highpass_filter(y,3.,dt,order=8)
    return y
