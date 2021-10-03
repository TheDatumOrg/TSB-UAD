from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

from tensorflow.keras.callbacks import EarlyStopping

# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
import numpy as np
# import time
# import pandas as pd
# import os
# import math
# from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler

class cnn:
#LSTM based

    # features is the # of time steps we want to predict
    def __init__(self, slidingwindow = 100, predict_time_steps=1, contamination = 0.1, epochs = 10, patience = 10, verbose=0):
        self.slidingwindow = slidingwindow
        self.predict_time_steps = predict_time_steps
        self.contamination = contamination
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.model_name = 'cnn'
        
    def fit(self, X_clean, X_dirty, ratio = 0.15):

        slidingwindow = self.slidingwindow
        predict_time_steps = self.predict_time_steps
        self.n_test_ = len(X_dirty)


        X_train, Y_train = self.create_dataset(X_clean, slidingwindow, predict_time_steps)
        
        X_test, Y_test = self.create_dataset(X_dirty, slidingwindow, predict_time_steps)


        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        #Generate model for predictor"""
        model = Sequential()
        
        # Convolutional Layer #1
        # Computes 32 features using a 1D filter(kernel) of with w with ReLU activation. 
        # Padding is added to preserve width.
        # Input Tensor Shape: [batch_size, w, 1] / batch_size = len(batch_sample)
        # Output Tensor Shape: [batch_size, w, num_filt_1] (num_filt_1 = 32 feature vectors)
        model.add(Conv1D(filters=8,
                         kernel_size=2,
                         strides=1,
                         padding='same',
                         activation='relu',
                         input_shape=(slidingwindow, 1)))
        
        
        model.add(MaxPooling1D(pool_size=2)) 

        model.add(Conv1D(filters=16,
                         kernel_size=2,
                         strides=1,
                         padding='valid',
                         activation='relu'))
        
        model.add(MaxPooling1D(pool_size=2))
                            #  strides=pool_strides_2, 
                            #  padding='valid'
                  
        model.add(Conv1D(filters=32,
                         kernel_size=2,
                         strides=1,
                         padding='valid',
                         activation='relu'))
        
        model.add(MaxPooling1D(pool_size=2))
        
        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 0.25 * w, num_filt_1 * num_filt_2]
        # Output Tensor Shape: [batch_size, 0.25 * w * num_filt_1 * num_filt_2]
        model.add(Flatten())
        
        # Dense Layer (Output layer)
        # Densely connected layer with 1024 neurons
        # Input Tensor Shape: [batch_size, 0.25 * w * num_filt_1 * num_filt_2]
        # Output Tensor Shape: [batch_size, 1024]
        model.add(Dense(units=64, activation='relu'))  
        
        # Dropout
        # Prevents overfitting in deep neural networks
        model.add(Dropout(rate=0.2))
        
        # Output layer
        # Input Tensor Shape: [batch_size, 1024]
        # Output Tensor Shape: [batch_size, p_w]
        model.add(Dense(units=predict_time_steps))
        
        model.compile(loss='mse', optimizer='adam')
        
        # Summarize model structure
        # model.summary()
        
        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=self.verbose, patience=self.patience)
        
        model.fit(X_train,Y_train,validation_split=ratio,
                  epochs=self.epochs,batch_size=64,verbose=self.verbose, callbacks=[es])

        # plt.figure()
        # plt.plot(model.history.history['loss'], label='train')
        # plt.plot(model.history.history['val_loss'], label='test')
        # plt.legend()

        
        prediction = model.predict(X_test)

        self.Y_test = Y_test
        self.prediction = prediction
        self.estimator = model
        self.n_initial = X_train.shape[0]
        
        return self
        
    def create_dataset(self, X, slidingwindow, predict_time_steps=1):  # X ia 1d array
        Xs, ys = [], []
        for i in range(len(X) - slidingwindow - predict_time_steps+1):
            tmp = X[i : i + slidingwindow + predict_time_steps]
            tmp= MinMaxScaler(feature_range=(0,1)).fit_transform(tmp.reshape(-1,1)).ravel()
            x = tmp[:slidingwindow]
            y = tmp[slidingwindow:]
            
            Xs.append(x)
            ys.append(y)
        return np.array(Xs), np.array(ys)
    

    
    def decision_function(self, X= False, measure = None):
        """Derive the decision score based on the given distance measure
        Parameters
        ----------
        X : numpy array of shape (n_samples, )
            The input samples.
        measure : object
            object for given distance measure with methods to derive the score
        Returns
        -------
        self : object
            Fitted estimator.
        """
        if type(X) != bool:
            self.X_train_ = X

        Y_test = self.Y_test

        score = []
        # measure.detector = self
        # measure.set_param()
        prediction = self.prediction

        for i in range(prediction.shape[0]):
            score.append(measure.measure(Y_test[i], prediction[i], 0))
        
        score = np.array(score)
        decision_scores_ = np.zeros(self.n_test_)
        
        decision_scores_[self.slidingwindow : (self.n_test_-self.predict_time_steps+1)]=score
        decision_scores_[: self.slidingwindow] = score[0]
        decision_scores_[self.n_test_-self.predict_time_steps+1:]=score[-1]
        
        self.decision_scores_ = decision_scores_
        # self._mu = np.mean(self.decision_scores_)
        # self._sigma = np.std(self.decision_scores_)
        return self