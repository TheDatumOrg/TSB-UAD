# Author: Yuhao Kang <yuhaok@uchicago.edu>

import numpy as np
from scipy import signal
from ..utils.metrics import metricor


def plotdata(data,label, ax, title=None, max_len=2000):
    range_anomaly = metricor().range_convers_new(label)
    ax.plot(data[:max_len],'k')
    for r in range_anomaly:
        if r[0]==r[1]:
            ax.plot(r[0],data[r[0]],'r.')
        else:
            ax.plot(range(r[0],r[1]+1),data[range(r[0],r[1]+1)],'r')
    ax.set_xlim([0,min(max_len, len(data))])
    ax.set_title(title)


def add_random_walk_trend(data, label, seed=5, p_walk=0.2):
    np.random.seed(seed)
    dims = 1
    step_n = len(data)-1
    step_set = [-1, 0, 1]
    origin = np.zeros((1,dims))
    # Simulate steps in 1D
    step_shape = (step_n,dims)
    steps = np.random.choice(a=step_set, size=step_shape)
    trend = np.concatenate([origin, steps]).cumsum(0)
    
    return np.ravel(trend)*np.std(data)*p_walk + data, label
   
def add_white_noise(data, label, seed=5, p_whitenoise=0.2):
    np.random.seed(int(seed))
    return data + np.random.normal(size=len(data))*np.std(data)*p_whitenoise, label
    

def filter_fft(data, label, p=21):

    # Convert data to freq domain and filter the high-freq component.

    n=len(data)
    
    t=np.linspace(-3,3,p)
    g=np.exp(-t**2)

    g_padded=np.hstack((g, np.zeros(n-p)))
    
    # fourier transform of signal
    xhat = np.fft.fft(data)
    
    # fourier transform of filter
    ghat = np.fft.fft(g_padded)
    
    # apply filter in fourier space
    tmp = xhat * ghat  
    
    # convert back to time domain
    return np.real(np.fft.ifft(tmp))/np.sum(g), label


def two_seg_freq_filter(data, label, p1=21, p2=51):
   
    # Split the data into two parts, apply two different Gaussian filters respectively and combine them back.
    
    l0 = len(data)//2
    data1 = data[:l0].copy()
    data2 = data[l0:].copy()
    
    data1_new, label1 =  filter_fft(data1, label[:l0], p=p1)
    data2_new, label2 =  filter_fft(data2, label[l0:], p=p2)
    
    return np.concatenate((data1_new,data2_new)), label



def select_region(length, contamination, period, seed):
    # given the total length, generate periods that will be transformed
    
    np.random.seed(int(seed))
    
    n = int(length*contamination)
    period = max(period, 10)
    
    m = int(n/period)
    
    region = []
    
    for i in range(m):
        s = int(length/m)*i
        e = int(length/m)*(i+1)
        r = np.random.choice(np.arange(s, e-period), size=1)
        region.append([int(r), int(r+period)])
        
    return region
    

def add_point_outlier(data, label, seed=5, outlierRatio=0.01):
    np.random.seed(int(seed))
    n = int(len(data) * outlierRatio)
    index = np.random.choice(len(data), n, replace=False)
    
    data_new = data.copy()
    data_new[index] = data_new[index] + 5*np.std(data)
    
    label_new = label.copy()
    
    label_new[index] = 1

    return data_new, label_new   


def flat_region(data, label, contamination, period, seed=5):
    # Replace with a flat region to the original data.
    region = select_region(len(data), contamination, period, seed)
    
    data_new = data.copy()
    label_new = label.copy()
    for r in region:
        data_new[r[0]:r[1]] = data[r[0]]
        label_new[r[0]:r[1]] = 1
        
    return data_new, label_new


def flip_segment(data, label, contamination, period, seed=5):
    # Flip a specific segment of the original time series.
    region = select_region(len(data), contamination, period, seed)

    data_new = data.copy()
    label_new = label.copy()  
    for r in region:
        data_new[r[0]:r[1]] = np.flipud(data[r[0]:r[1]])
        label_new[r[0]:r[1]] = 1
    
    return data_new, label_new



def change_segment_add_scale(data, label, contamination, period, seed=5, para=2, method='scale'):
    region = select_region(len(data), contamination, period, seed)

    data_new = data.copy()
    label_new = label.copy()
    if method=='scale':  
        for r in region:
            data_new[r[0]:r[1]]  *= para
            label_new[r[0]:r[1]] = 1
        
    elif method == 'add':
        for r in region:
            data_new[r[0]:r[1]]  += para
            label_new[r[0]:r[1]] = 1

    return data_new, label_new


def change_segment_normalization(data, label, contamination, period, seed=5, method='Z-score'):
    region = select_region(len(data), contamination, period, seed)

    data_new = data.copy()
    label_new = label.copy()
    
    for r in region:

        x = data[r[0]:r[1]].copy()

    
        if method=='Z-score':
            x = (x - np.mean(x))/np.std(x)
        elif method=='Min-max':
            x = (x - np.min(x))/(np.max(x)-np.min(x))
        elif method=='MeanNorm':
            x = (x - np.mean(x))/(np.max(x)-np.min(x))
        elif method=='MedianNorm':
            x /= np.median(x)
        elif method=='UnitLength':
            x /= np.sum(x**2)**0.5
        elif method=='Logistic':
            x = 1/(1+np.exp(-x))
        elif method=='Tanh':
            y= np.exp(2*x)
            x = (y-1)/(y+1)
        
        data_new[r[0]:r[1]] = x
        label_new[r[0]:r[1]] = 1
        
    return data_new, label_new


def change_segment_partial(data, label, contamination, period, seed=5, ratio=0.2, method='flat'):
    region = select_region(len(data), contamination, period, seed)

    data_new = data.copy()
    label_new = label.copy()
    
    for r in region:
        x = data[r[0]:r[1]].copy()
    
        l0 = int(len(x)*ratio)
        if method=='flat':
            x[:l0] = x[0]
        elif method=='noise':
            x[:l0] += np.random.normal(size=l0)*np.std(x)*0.3
            
        data_new[r[0]:r[1]] = x
        label_new[r[0]:(r[0]+l0)] = 1

    return data_new, label_new



def change_segment_resampling(data, label, contamination, period, seed, resampling_ratio):
    region = select_region(len(data), contamination, period, seed)
    
    pre = 0
    tmp_data = []
    tmp_label = []
    for r in region:
        tmp_data += list(data[pre:r[0]])
        tmp_label += list(label[pre:r[0]])
        x = data[r[0]:r[1]].copy()
    
        
        length_new = int((r[1]-r[0])*resampling_ratio)
        f = signal.resample(x, length_new)
        
        tmp_data += (list(f))
        tmp_label += ([1]*len(f))
        
        pre = r[1]
        
    tmp_data += list(data[pre:len(data)])
    tmp_label += list(label[pre:len(data)])
    
    return np.array(tmp_data), np.array(tmp_label)


class transform():
    def __init__(self, kind = 1):
        self.kind = kind

    def transform(self, data, label, contamination=None, period=None, para=None, method=None, seed=5):
        '''

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        label : TYPE
            DESCRIPTION.
        period : TYPE, optional
            DESCRIPTION. The default is None.
        seed : TYPE, optional
            DESCRIPTION. The default is 5.
        para : TYPE
            kind 1: p_walk=para
            kind 2: outlierRatio=para
            kind 3: p=para
            kind 4: p1=para[0], p2=para[1]
            kind 5: p_whitenoise=para
            kind 8: vary_strength=para
            kind 9: para=para
            kind 11: ratio=para
            kind 12: resampling_ratio=para

        '''
        trans_name=''
        if self.kind==1:        
            data, label = add_random_walk_trend(data, label, seed=seed, p_walk=para)
            trans_name='_add_random_walk_trend'
        elif self.kind==2:             
            data, label = add_point_outlier(data,label,seed=seed, outlierRatio=para)
            trans_name='_add_point_outlier'
        elif self.kind==3:      
            data, label = filter_fft(data, label, p=para)
            trans_name='_filter_fft'
        elif self.kind==4:      
            data, label = two_seg_freq_filter(data, label, p1=para[0], p2=para[1])
            trans_name='_two_seg_freq_filter'
        elif self.kind==5:   
            data, label = add_white_noise(data, label, seed=seed, p_whitenoise=para)
            trans_name='_add_white_noise'
        elif self.kind==6:   
            data, label = flat_region(data, label, contamination, period, seed)
            trans_name='_flat_region'
        elif self.kind==7:   
            data, label = flip_segment(data, label, contamination, period, seed)
            trans_name='flip_segment'
        elif self.kind==9:   
            data, label = change_segment_add_scale(data, label,contamination, period, seed, para=para, method=method)
            trans_name='_change_segment_add_scale'
        elif self.kind==10:  
            data, label = change_segment_normalization(data, label, contamination, period, seed, method=method)
            trans_name='_change_segment_normalization'
        elif self.kind==11:   
             
            data, label = change_segment_partial(data, label, contamination, period, seed, ratio=para, method=method)
            trans_name='_change_segment_partial'
        elif self.kind==12:  
            data, label = change_segment_resampling(data, label, contamination, period, seed, resampling_ratio=para)
            trans_name='_change_segment_resampling'
            
        self.data = data
        self.label = label
        self.trans_name = trans_name
        return self