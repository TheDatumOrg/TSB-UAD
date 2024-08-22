from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping


import numpy as np
import math
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler

class lstm:
    """
    Implementation of LSTM-AD
    
    Parameters
    ----------
    slidingwindow : int
        Subsequence length to analyze.
    predict_time_steps : int, (default=1)
        The length of the subsequence to predict.
    epochs : int, (default=10)
        Number of epochs for the training phase
    patience : int, (default=10)
        Number of epoch to wait before early stopping during training

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples - subsequence_length,)
        The anomaly score.
        The higher, the more abnormal. Anomalies tend to have higher
        scores. This value is available once decision_function is called.
    """

    def __init__(self, slidingwindow = 100, predict_time_steps=1, epochs = 10, patience = 10, verbose=0):
        self.slidingwindow = slidingwindow
        self.predict_time_steps = predict_time_steps
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.model_name = 'LSTM2'
        
    def fit(self, X_clean, X_dirty, ratio = 0.15):
        """Fit detector.
        
        Parameters
        ----------
        X_clean : numpy array of shape (n_samples, )
            The input training samples.
        X_dirty : numpy array of shape (n_samples, )
            The input testing samples.
        ratio : flaot, ([0,1])
            The ratio for the train validation split
        
        Returns
        -------
        self : object
            Fitted estimator.
        """

        slidingwindow = self.slidingwindow
        predict_time_steps = self.predict_time_steps
        self.n_test_ = len(X_dirty)

        X_train, Y_train = self.create_dataset(X_clean, slidingwindow, predict_time_steps)
        X_test, Y_test = self.create_dataset(X_dirty, slidingwindow, predict_time_steps)
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        model = Sequential()
        model.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])))
        model.add(LSTM(50))
        model.add(Dense(predict_time_steps))
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=self.verbose, patience=self.patience)
        
        model.fit(X_train,Y_train,validation_split=ratio,
                  epochs=self.epochs,batch_size=64,verbose=self.verbose, callbacks=[es])
        
        prediction = model.predict(X_test)

        self.Y = Y_test
        self.estimation = prediction
        self.estimator = model
        self.n_initial = X_train.shape[0]
        
        return self
        
    def create_dataset(self, X, slidingwindow, predict_time_steps=1): 
        Xs, ys = [], []
        for i in range(len(X) - slidingwindow - predict_time_steps+1):
            tmp = X[i : i + slidingwindow + predict_time_steps]
            tmp= MinMaxScaler(feature_range=(0,1)).fit_transform(tmp.reshape(-1,1)).ravel()
            x = tmp[:slidingwindow]
            y = tmp[slidingwindow:]
            
            Xs.append(x)
            ys.append(y)
        return np.array(Xs), np.array(ys)
    

    
    def decision_function(self, measure = None):
        """Derive the decision score based on the given distance measure
        
        Parameters
        ----------
        measure : object
            object for given distance measure with methods to derive the score
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
       
        Y_test = self.Y

        score = np.zeros(self.n_test_)
        estimation = self.estimation

        for i in range(estimation.shape[0]):
            score[i - estimation.shape[0]] = measure.measure(Y_test[i], estimation[i], self.n_test_ - estimation.shape[0] + i)

        score[0: - estimation.shape[0]] = score[- estimation.shape[0]]
        
        self.decision_scores_ = score
        return self

