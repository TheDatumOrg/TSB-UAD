from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np



def find_length(data):
    """
    determine sliding window (period) based on autocorrelation.
        
    Parameters
    ----------
    data : numpy array of shape (n_samples, )
        The time series on which we find the optimal subsequence length.
    
    Returns
    -------
    length : int
        argmax on the autocorrelation curve. Cannot be smaller than 3 and bigger than 300.
        In case of extreme small (below 3) or big (above 300) argmax, we set a default subseuqence length and return 100. 
    """
    if len(data.shape)>1:
        return 0
    data = data[:min(20000, len(data))]
    
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    
    
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 100
        return local_max[max_local_max]+base
    except:
        return 100
    
