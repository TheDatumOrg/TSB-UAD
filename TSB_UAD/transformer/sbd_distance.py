#  SBD distance

import numpy as np
from numpy.linalg import norm
from numpy.fft import fft, ifft

def roll_zeropad(a, shift, axis=None):
    a = np.asanyarray(a)
    if shift == 0:
        return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift, n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res
    
def _ncc_c(x, y):
    """
    >>> _ncc_c([1,2,3,4], [1,2,3,4])
    array([ 0.13333333,  0.36666667,  0.66666667,  1.        ,  0.66666667,
            0.36666667,  0.13333333])
    >>> _ncc_c([1,1,1], [1,1,1])
    array([ 0.33333333,  0.66666667,  1.        ,  0.66666667,  0.33333333])
    >>> _ncc_c([1,2,3], [-1,-1,-1])
    array([-0.15430335, -0.46291005, -0.9258201 , -0.77151675, -0.46291005])
    """
    den = np.array(norm(x) * norm(y))
    den[den == 0] = np.Inf

    x_len = len(x)
    fft_size = 1 << (2*x_len-1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
    return np.real(cc) / den

def _sbd(x, y):
    """
    >>> _sbd([1,1,1], [1,1,1])
    (-2.2204460492503131e-16, array([1, 1, 1]))
    >>> _sbd([0,1,2], [1,2,3])
    (0.043817112532485103, array([1, 2, 3]))
    >>> _sbd([1,2,3], [0,1,2])
    (0.043817112532485103, array([0, 1, 2]))
    """
    ncc = _ncc_c(x, y)
    idx = ncc.argmax()
    dist = 1 - ncc[idx]
    yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))

    return dist, yshift