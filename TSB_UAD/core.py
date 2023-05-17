import os
import math
import numpy as np
import pandas as pd
from TSB_UAD.models.distance import Fourier
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.metrics import metricor
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics
from sklearn.preprocessing import MinMaxScaler


def tsb_uad(data, label, model, slidingWindow=None, metric='all'):
    # Training and execution of the model
    if model == 'IForest':
        from TSB_UAD.models.iforest import IForest
        
        if slidingWindow:
            X_data = Window(window = slidingWindow).convert(data).to_numpy()
        else:
            slidingWindow = find_length(data)
            X_data = Window(window = slidingWindow).convert(data).to_numpy()

        clf = IForest(n_jobs=1)
        x = X_data
        clf.fit(X_data)
        score = clf.decision_scores_
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
        score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

    elif model == 'DAMP':
        from TSB_UAD.models.damp import DAMP
        
        if slidingWindow:
            slidingWindow = slidingWindow
        else:
            slidingWindow = find_length(data)

        clf = DAMP(m = slidingWindow,sp_index=slidingWindow+1)
        x = data
        clf.fit(x)
        score = clf.decision_scores_
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
        score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

    elif model == 'SAND':
        from TSB_UAD.models.sand import SAND

        if slidingWindow:
            slidingWindow = slidingWindow
        else:
            slidingWindow = find_length(data)

        clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
        x = data
        clf.fit(x,overlaping_rate=int(1.5*slidingWindow))
        score = clf.decision_scores_
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

    elif model == 'Series2Graph':
        from TSB_UAD.models.series2graph import Series2Graph
        
        if slidingWindow:
            slidingWindow = slidingWindow
        else:
            slidingWindow = find_length(data)

        print("slidingWindow: ", slidingWindow)
        s2g = Series2Graph(pattern_length=slidingWindow)
        s2g.fit(data)
        query_length = 2*slidingWindow
        s2g.score(query_length=query_length,dataset=data)

        score = s2g.decision_scores_
        score = np.array([score[0]]*math.ceil(query_length//2) + list(score) + [score[-1]]*(query_length//2))
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

    elif model == 'LOF':
        from TSB_UAD.models.lof import LOF

        if slidingWindow:
            X_data = Window(window = slidingWindow).convert(data).to_numpy()
        else:
            slidingWindow = find_length(data)
            X_data = Window(window = slidingWindow).convert(data).to_numpy()

        clf = LOF(n_neighbors=20, n_jobs=-1)
        x = X_data
        clf.fit(x)
        score = clf.decision_scores_
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
        score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

    elif model == 'MatrixProfile':
        from TSB_UAD.models.matrix_profile import MatrixProfile

        if slidingWindow:
            slidingWindow = slidingWindow
        else:
            slidingWindow = find_length(data)

        clf = MatrixProfile(window = slidingWindow)
        x = data
        clf.fit(x)
        score = clf.decision_scores_
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
        score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

    elif model == 'NORMA':
        from TSB_UAD.models.norma import NORMA
        
        if slidingWindow:
            slidingWindow = slidingWindow
        else:
            slidingWindow = find_length(data)

        clf = NORMA(pattern_length = slidingWindow, nm_size=3*slidingWindow)
        x = data
        clf.fit(x)
        score = clf.decision_scores_
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
        score = np.array([score[0]]*((slidingWindow-1)//2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
        
    elif model == 'PCA':
        from TSB_UAD.models.pca import PCA

        if slidingWindow:
            X_data = Window(window = slidingWindow).convert(data).to_numpy()
        else:
            slidingWindow = find_length(data)
            X_data = Window(window = slidingWindow).convert(data).to_numpy()
        
        clf = PCA()
        x = X_data
        clf.fit(x)
        score = clf.decision_scores_
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
        score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

    elif model == 'POLY':
        from TSB_UAD.models.poly import POLY

        if slidingWindow:
            slidingWindow = slidingWindow
        else:
            slidingWindow = find_length(data)

        clf = POLY(power=3, window = slidingWindow)
        x = data
        clf.fit(x)
        measure = Fourier()
        measure.detector = clf
        measure.set_param()
        clf.decision_function(measure=measure)
        score = clf.decision_scores_
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
        
    elif model == 'OCSVM':
        from TSB_UAD.models.ocsvm import OCSVM
        
        if slidingWindow:
            slidingWindow = slidingWindow
        else:
            slidingWindow = find_length(data)
        
        data_train = data[:int(0.1*len(data))]
        data_test = data
        X_train = Window(window = slidingWindow).convert(data_train).to_numpy()
        X_test = Window(window = slidingWindow).convert(data_test).to_numpy()

        modelName='OCSVM'
        X_train_ = MinMaxScaler(feature_range=(0,1)).fit_transform(X_train.T).T
        X_test_ = MinMaxScaler(feature_range=(0,1)).fit_transform(X_test.T).T
        clf = OCSVM(nu=0.05)
        clf.fit(X_train_, X_test_)
        score = clf.decision_scores_
        score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

    elif model == 'LSTM':
        from TSB_UAD.models.lstm import lstm

        if slidingWindow:
            slidingWindow = slidingWindow
        else:
            slidingWindow = find_length(data)

        data_train = data[:int(0.1*len(data))]
        data_test = data
        X_train = Window(window = slidingWindow).convert(data_train).to_numpy()
        X_test = Window(window = slidingWindow).convert(data_test).to_numpy()
            
        modelName='LSTM'
        clf = lstm(slidingwindow = slidingWindow, predict_time_steps=1, epochs = 50, patience = 5, verbose=0)
        clf.fit(data_train, data_test)
        measure = Fourier()
        measure.detector = clf
        measure.set_param()
        clf.decision_function(measure=measure)
        score = clf.decision_scores_
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel() 

    elif model == 'AE':
        from TSB_UAD.models.AE_mlp2 import AE_MLP2

        if slidingWindow:
            slidingWindow = slidingWindow
        else:
            slidingWindow = find_length(data)

        data_train = data[:int(0.1*len(data))]
        data_test = data
        X_train = Window(window = slidingWindow).convert(data_train).to_numpy()
        X_test = Window(window = slidingWindow).convert(data_test).to_numpy()
        
        modelName='AE'
        clf = AE_MLP2(slidingWindow = slidingWindow, epochs=100, verbose=0)
        clf.fit(data_train, data_test)        
        score = clf.decision_scores_
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

    elif model == 'CNN':
        from TSB_UAD.models.cnn import cnn

        if slidingWindow:
            slidingWindow = slidingWindow
        else:
            slidingWindow = find_length(data)

        data_train = data[:int(0.1*len(data))]
        data_test = data
        X_train = Window(window = slidingWindow).convert(data_train).to_numpy()
        X_test = Window(window = slidingWindow).convert(data_test).to_numpy()

        modelName='CNN'
        clf = cnn(slidingwindow = slidingWindow, predict_time_steps=1, epochs = 100, patience = 5, verbose=0)
        clf.fit(data_train, data_test)
        measure = Fourier()
        measure.detector = clf
        measure.set_param()
        clf.decision_function(measure=measure)
        score = clf.decision_scores_
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()


    results = get_metrics(score, label, metric=metric, slidingWindow=slidingWindow)
    return results

