import numpy as np
import math
from models.distance import Fourier
from vus.metrics import get_metrics
from utils.slidingWindows import find_length, find_length_rank
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from functools import wraps
import time
import os
import logging

# import sys
# sys.path.append('..')
# from models.NormA import NORMA

from TSB_UAD.models.LOF import LOF
from TSB_UAD.models.IForest import IForest
from TSB_UAD.models.POLY import POLY
from TSB_UAD.models.MatrixProfile import MatrixProfile
from TSB_UAD.models.PCA import PCA
from TSB_UAD.models.HBOS import HBOS
from TSB_UAD.models.OCSVM import OCSVM
from TSB_UAD.models.AE import AutoEncoder
from TSB_UAD.models.CNN import CNN
from TSB_UAD.models.LSTM import LSTM
from TSB_UAD.models.TranAD import TranAD

def run_iforest_dev(data, periodicity, n_estimators, n_jobs=1):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_lof_dev(data, periodicity, n_neighbors, n_jobs=1):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = LOF(slidingWindow=slidingWindow, n_neighbors=n_neighbors, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_matrix_profile_dev(data, periodicity, n_jobs=1):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = MatrixProfile(slidingWindow = slidingWindow, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_pca_dev(data, periodicity, n_jobs=1):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = PCA(slidingWindow = slidingWindow)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_norma_dev(data, periodicity, clustering, n_jobs=1):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = NORMA(pattern_length=slidingWindow, nm_size=3*slidingWindow, clustering=clustering)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
    if len(score) > len(data):
        start = len(score) - len(data)
        score = score[start:]
    return score

def run_hbos_dev(data, periodicity, n_jobs=1):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = HBOS()
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_poly_dev(data, periodicity, power, n_jobs=1):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = POLY(power=power, window = slidingWindow)
    clf.fit(data)
    measure = Fourier()
    measure.detector = clf
    measure.set_param()
    clf.decision_function(measure=measure)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_ocsvm_dev(data, periodicity, n_jobs=1):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = OCSVM()
    clf.fit(data[:int(0.1*len(data)+slidingWindow)])
    score = clf.decision_function(data)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

# def run_ae_dev(data, periodicity, hidden_neurons, n_jobs=1):
#     slidingWindow = find_length_rank(data, rank=periodicity)
#     clf = AutoEncoder(hidden_neurons=hidden_neurons, batch_size=16)

#     X_train = Window(window = slidingWindow).convert(data[:int(0.1*len(data)+slidingWindow)]).to_numpy()
#     X_test = Window(window = slidingWindow).convert(data).to_numpy()
#     X_train_ = MinMaxScaler(feature_range=(0,1)).fit_transform(X_train.T).T
#     X_test_ = MinMaxScaler(feature_range=(0,1)).fit_transform(X_test.T).T

#     clf.fit(X_train_)
#     score = clf.decision_function(X_test_)
#     score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
#     score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
#     return score

def run_ae_dev(data, periodicity, hidden_neurons, n_jobs=1):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = AutoEncoder(slidingWindow=slidingWindow, hidden_neurons=hidden_neurons, batch_size=16, epochs=10)
    clf.fit(data[:int(0.1*len(data)+slidingWindow)])
    score = clf.decision_function(data)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_cnn_dev(data, periodicity, num_channel, activation, n_jobs=1):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = CNN(slidingWindow=slidingWindow, num_channel=num_channel)
    data_train = data[:int(0.1*len(data)+slidingWindow)]
    clf.fit(data_train)

    measure = Fourier()
    measure.detector = clf
    measure.set_param()
    score = clf.decision_function(data, measure=measure)
    # score = clf.decision_function(data)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_lstm_dev(data, periodicity, hidden_dim, activation, n_jobs=1):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = LSTM(slidingWindow=slidingWindow, hidden_dim=hidden_dim, epochs=10)
    data_train = data[:int(0.3*len(data)+slidingWindow)]
    clf.fit(data_train)

    measure = Fourier()
    measure.detector = clf
    measure.set_param()
    score = clf.decision_function(data, measure=measure)
    # score = clf.decision_function(data)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_TranAD(data, periodicity):
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = TranAD(slidingWindow=10)
    # clf = TranAD(slidingWindow=slidingWindow, epochs=5)
    clf.fit(data[:int(0.3*len(data)+slidingWindow)])
    score = clf.decision_function(data)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score