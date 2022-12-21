import numpy as np
# from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler
# import math
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers


class AE_MLP:
#Autoencoder using LSTM    
    def __init__(self, slidingWindow = 100,  contamination = 0.1, epochs = 10):
        self.slidingWindow = slidingWindow
        self.contamination = contamination
        self.epochs = epochs
        self.model_name = 'AE_MLP'

    def fit(self, X_clean, X_dirty, ratio = 0.15):

        TIME_STEPS =  self.slidingWindow
        epochs = self.epochs
        

        # self.n_train_ = len(X_dirty)
        

        X_train = self.create_dataset(X_clean,TIME_STEPS)
        X_test = self.create_dataset(X_dirty,TIME_STEPS)
        
        X_train = MinMaxScaler().fit_transform(X_train.T).T
        X_test = MinMaxScaler().fit_transform(X_test.T).T

        model = Sequential()
        model.add(layers.Dense(32,  activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(TIME_STEPS, activation='relu'))
 
        model.compile(optimizer='adam', loss='mse')
        
        
        history = model.fit(X_train, X_train,
                        epochs=epochs,
                        batch_size=64,
                        shuffle=False,
                        validation_split=0.15,
                        callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")])
        
        # model.summary()
        
        # plt.figure()
        # plt.plot(history.history["loss"],'r')
        # plt.plot(history.history["val_loss"],'b')

        test_predict = model.predict(X_test)
        test_mae_loss = np.mean(np.abs(test_predict - X_test), axis=1)
        nor_test_mae_loss = MinMaxScaler().fit_transform(test_mae_loss.reshape(-1,1)).ravel()
        score = np.zeros(len(X_dirty))
        score[self.slidingWindow//2:self.slidingWindow//2+len(test_mae_loss)]=nor_test_mae_loss
        score[:self.slidingWindow//2]=nor_test_mae_loss[0]
        score[self.slidingWindow//2+len(test_mae_loss):]=nor_test_mae_loss[-1]
        
        self.decision_scores_ = score
        
        return self
        
    
    # Generated training sequences for use in the model.
    def create_dataset(self, X, time_steps):
        output = []
        for i in range(len(X) - time_steps + 1):
            output.append(X[i : (i + time_steps)])
        return np.stack(output)

# =============================================================================
#     def decision_function(self, X= False, measure = None):
#         """Derive the decision score based on the given distance measure
#         Parameters
#         ----------
#         X : numpy array of shape (n_samples, )
#             The input samples.
#         measure : object
#             object for given distance measure with methods to derive the score
#         Returns
#         -------
#         self : object
#             Fitted estimator.
#         """
#         if type(X) != bool:
#             self.X_train_ = X
#         n_train_ = self.n_train_
#         # self.neighborhood = n_train_
#         # autoencoder = self.estimator
#         X_test = self.X_test
#         slidingWindow = self.slidingWindow
#         # score = np.zeros(n_train_)
#         measure.detector = self
#         measure.set_param()
#         estimation = self.estimation
#         n_initial = self.n_initial
# 
#         # i = 0
#         # m = 0
#         # for i in range(estimation.shape[0]):
#             # score[i + n_initial + window] = measure.measure(X_test[i], estimation[i], i + n_initial + window)
#         score = np.mean(np.abs(estimation-X_test), axis=1).ravel()
#         score = np.array([score[0]]*math.ceil((slidingWindow+1)/2) + list(score) + [score[-1]]*((slidingWindow+1)//2))
#         print('done!')
#         self.decision_scores_ = score
#         # self._mu = np.mean(self.decision_scores_)
#         # self._sigma = np.std(self.decision_scores_)
#         return self
# 
#     def predict_proba(self, X, method='linear', measure = None):
#         """Predict the probability of a sample being outlier. Two approaches
#         are possible:
#         1. simply use Min-max conversion to linearly transform the outlier
#            scores into the range of [0,1]. The model must be
#            fitted first.
#         2. use unifying scores, see :cite:`kriegel2011interpreting`.
#         Parameters
#         ----------
#         X : numpy array of shape (n_samples, n_features)
#             The input samples.
#         method : str, optional (default='linear')
#             probability conversion method. It must be one of
#             'linear' or 'unify'.
#         Returns
#         -------
#         outlier_probability : numpy array of shape (n_samples,)
#             For each observation, tells whether or not
#             it should be considered as an outlier according to the
#             fitted model. Return the outlier probability, ranging
#             in [0,1].
#         """
# 
#         check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
#         train_scores = self.decision_scores_
# 
#         self.fit(X)
#         self.decision_function(measure = measure)
#         test_scores = self.decision_scores_
# 
#         probs = np.zeros([X.shape[0], int(self._classes)])
#         if method == 'linear':
#             scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
#             probs[:, 1] = scaler.transform(
#                 test_scores.reshape(-1, 1)).ravel().clip(0, 1)
#             probs[:, 0] = 1 - probs[:, 1]
#             return probs
# 
#         elif method == 'unify':
#             # turn output into probability
#             pre_erf_score = (test_scores - self._mu) / (
#                     self._sigma * np.sqrt(2))
#             erf_score = math.erf(pre_erf_score)
#             probs[:, 1] = erf_score.clip(0, 1).ravel()
#             probs[:, 0] = 1 - probs[:, 1]
#             return probs
#         else:
#             raise ValueError(method,
#                              'is not a valid probability conversion method')
# =============================================================================
