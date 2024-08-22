from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np

import matplotlib.patches as mpatches 
import matplotlib.pyplot as plt

from ..vus.utils.metrics import metricor

def plotFig(data, label, score, slidingWindow, fileName, modelName, plotRange=None):
    grader = metricor()
    
    range_anomaly = grader.range_convers_new(label)
    
    max_length = len(score)

    if plotRange==None:
        plotRange = [0,max_length]
    
    fig3 = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig3.add_gridspec(3, 4)
    
    
    f3_ax1 = fig3.add_subplot(gs[0, :-1])
    plt.tick_params(labelbottom=False)

    plt.plot(data[:max_length],'k')
    for r in range_anomaly:
        if r[0]==r[1]:
            plt.plot(r[0],data[r[0]],'r.')
        else:
            plt.plot(range(r[0],r[1]+1),data[range(r[0],r[1]+1)],'r')
    
    plt.xlim(plotRange)
    

    f3_ax2 = fig3.add_subplot(gs[1, :-1])
    
    plt.plot(score[:max_length])
    plt.hlines(np.mean(score)+3*np.std(score),0,max_length,linestyles='--',color='red')
    plt.ylabel('score')
    
    plt.xlim(plotRange)
    
    
    #plot the data
    f3_ax3 = fig3.add_subplot(gs[2, :-1])
    index = ( label + 2*(score > (np.mean(score)+3*np.std(score))))
    cf = lambda x: 'k' if x==0 else ('r' if x == 1 else ('g' if x == 2 else 'b') )
    cf = np.vectorize(cf)
    
    color = cf(index[:max_length])
    black_patch = mpatches.Patch(color = 'black', label = 'TN')
    red_patch = mpatches.Patch(color = 'red', label = 'FN')
    green_patch = mpatches.Patch(color = 'green', label = 'FP')
    blue_patch = mpatches.Patch(color = 'blue', label = 'TP')
    plt.scatter(np.arange(max_length), data[:max_length], c=color, marker='.')
    plt.legend(handles = [black_patch, red_patch, green_patch, blue_patch], loc= 'best')
    plt.xlim(plotRange)
    
        