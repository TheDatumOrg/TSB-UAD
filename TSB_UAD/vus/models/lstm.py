from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping

# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler

class lstm:
#LSTM based

    # features is the # of time steps we want to predict
    def __init__(self, slidingwindow = 100, predict_time_steps=1, contamination = 0.1, epochs = 10, patience = 10, verbose=0):
        self.slidingwindow = slidingwindow
        self.predict_time_steps = predict_time_steps
        self.contamination = contamination
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.model_name = 'LSTM2'
        
    def fit(self, X_clean, X_dirty, ratio = 0.15):

        slidingwindow = self.slidingwindow
        predict_time_steps = self.predict_time_steps
        self.n_test_ = len(X_dirty)


        X_train, Y_train = self.create_dataset(X_clean, slidingwindow, predict_time_steps)
        
        X_test, Y_test = self.create_dataset(X_dirty, slidingwindow, predict_time_steps)


        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        model = Sequential()
        # model.add(LSTM(units=50,input_shape=(X_train.shape[1], X_train.shape[2])))
        # model.add(LSTM(50))
        # model.add(LSTM(50))
        # model.add(Dense(1))
        model.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])))
        # model.add(LSTM(50,return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(predict_time_steps))
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        
        # model.summary()
        
        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=self.verbose, patience=self.patience)
        
        model.fit(X_train,Y_train,validation_split=ratio,
                  epochs=self.epochs,batch_size=64,verbose=self.verbose, callbacks=[es])


        
        prediction = model.predict(X_test)

        self.Y = Y_test
        self.estimation = prediction
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
        n_test_ = self.n_test_
        # self.neighborhood = n_train_

        Y_test = self.Y

        score = np.zeros(n_test_)
        # measure.detector = self
        # measure.set_param()
        estimation = self.estimation

        for i in range(estimation.shape[0]):
            

            score[i - estimation.shape[0]] = measure.measure(Y_test[i], estimation[i], n_test_ - estimation.shape[0] + i)
            # score[i - estimation.shape[0]] = np.abs(Y_test[i] - estimation[i])
        score[0: - estimation.shape[0]] = score[- estimation.shape[0]]
        
        self.decision_scores_ = score
        # self._mu = np.mean(self.decision_scores_)
        # self._sigma = np.std(self.decision_scores_)
        return self

    def predict_proba(self, X, method='linear', measure = None):
        """Predict the probability of a sample being outlier. Two approaches
        are possible:
        1. simply use Min-max conversion to linearly transform the outlier
           scores into the range of [0,1]. The model must be
           fitted first.
        2. use unifying scores, see :cite:`kriegel2011interpreting`.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        method : str, optional (default='linear')
            probability conversion method. It must be one of
            'linear' or 'unify'.
        Returns
        -------
        outlier_probability : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. Return the outlier probability, ranging
            in [0,1].
        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        train_scores = self.decision_scores_

        self.fit(X)
        self.decision_function(measure = measure)
        test_scores = self.decision_scores_

        probs = np.zeros([X.shape[0], int(self._classes)])
        if method == 'linear':
            scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
            probs[:, 1] = scaler.transform(
                test_scores.reshape(-1, 1)).ravel().clip(0, 1)
            probs[:, 0] = 1 - probs[:, 1]
            return probs

        elif method == 'unify':
            # turn output into probability
            pre_erf_score = (test_scores - self._mu) / (
                    self._sigma * np.sqrt(2))
            erf_score = math.erf(pre_erf_score)
            probs[:, 1] = erf_score.clip(0, 1).ravel()
            probs[:, 0] = 1 - probs[:, 1]
            return probs
        else:
            raise ValueError(method,
                             'is not a valid probability conversion method')
