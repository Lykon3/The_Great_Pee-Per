import numpy as np
from scipy.stats import entropy

def rolling_entropy(signal, window=50):
    """
    Calculate rolling Shannon entropy over a time series.
    Returns an array of entropy values.
    """
    ent = []
    for i in range(len(signal) - window + 1):
        window_data = signal[i:i + window]
        hist, _ = np.histogram(window_data, bins='auto', density=True)
        hist = hist[hist > 0]
        ent.append(entropy(hist))
    return np.array(ent)
