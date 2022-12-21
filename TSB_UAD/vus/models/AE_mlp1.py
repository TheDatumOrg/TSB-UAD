import numpy as np
# from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers


class AE_MLP1:
#Autoencoder using LSTM    
    def __init__(self, slidingWindow = 100,  contamination = 0.1, epochs = 10):
        self.slidingWindow = slidingWindow
        self.contamination = contamination
        self.epochs = epochs
        self.model_name = 'AE_MLP1'

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
        # model.add(layers.Dense(16, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(8, activation='relu'))
        # model.add(layers.Dense(16, activation='relu'))
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