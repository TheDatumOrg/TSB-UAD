import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from TSB_AD.models.distance import Fourier
from TSB_AD.models.feature import Window
from TSB_AD.utils.slidingWindows import find_length,plotFig, printResult
from sklearn.preprocessing import MinMaxScaler

#%%  prepare data

filepath = '../data/benchmark/ECG/MBA_ECG805_data.out'
df = pd.read_csv(filepath, header=None).to_numpy()

name = filepath.split('/')[-1]
max_length = 10000

data = df[:max_length,0].astype(float)
label = df[:max_length,1]
    
slidingWindow = find_length(data)
X_data = Window(window = slidingWindow).convert(data).to_numpy()


data_train = data[:int(0.1*len(data))]
data_test = data

X_train = Window(window = slidingWindow).convert(data_train).to_numpy()
X_test = Window(window = slidingWindow).convert(data_test).to_numpy()
#%%
from TSB_AD.models.iforest import IForest

modelName='IForest'
clf = IForest(n_jobs=1)
x = X_data
clf.fit(x)
score = clf.decision_scores_
score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

# save result as figure
plotFig(data, label, score, slidingWindow, fileName=name, modelName=modelName) #, plotRange=[1775,2200]
plt.savefig('../result/'+modelName+'.png')
plt.close()

# save evaluation parameter to file
output=printResult(data, label, score, slidingWindow, fileName=name, modelName=modelName)
f = open('../result/'+modelName+'.txt', "w")
for item in output[:-1]:
    f.write(str(round(item,2)))
    f.write(',')
f.write(str(round(output[-1],2)))
f.close()
