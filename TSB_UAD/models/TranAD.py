#  This function is adapted from [TranAD] by [imperial-qore]
#  Original source: [https://github.com/imperial-qore/TranAD]

from __future__ import division
from __future__ import print_function

import numpy as np
import math
import torch
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from sklearn.preprocessing import MinMaxScaler
from .base import BaseDetector
from .feature import Window
from ..utils.stat_models import pairwise_distances_no_broadcast
from ..utils.dataset import TSDataset
from ..utils.utility import get_activation_by_name
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model)
        )
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos : pos + x.size(0), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, *args, **kwargs):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, *args, **kwargs):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class TranADModel(nn.Module):
    def __init__(self, batch=128, feats=1, slidingWindow=10):
        super(TranADModel, self).__init__()
        self.name = "TranAD"
        self.batch = batch
        self.n_feats = feats
        self.n_window = slidingWindow
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2

class TranAD(BaseDetector):

    def __init__(self,
                 slidingWindow=10,
                 feats=1,
                 learning_rate=1e-4,
                 epochs=5,
                 batch_size=128,
                 dropout_rate=0.2,
                 weight_decay=1e-5,
                 # validation_size=0.1,
                 preprocessing=False,
                 loss_fn=None,
                 verbose=False,
                 # random_state=None,
                 contamination=0.1,
                 device=None):
        super(TranAD, self).__init__(contamination=contamination)

        # save the initialization values
        self.slidingWindow = slidingWindow
        self.feats = feats
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.preprocessing = preprocessing
        self.loss_fn = loss_fn
        self.verbose = verbose
        self.device = device

        # create default loss functions
        if self.loss_fn is None:
            self.loss_fn = torch.nn.MSELoss()

        # create default calculation device (support GPU if available)
        if self.device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

    # noinspection PyUnresolvedReferences
    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        n_samples, n_features = X.shape

        if n_features == 1: 
            # Converting time series data into matrix format
            X = Window(window = self.slidingWindow).convert(X).to_numpy()

        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        # conduct standardization if needed
        if self.preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            train_set = TSDataset(X=X, mean=self.mean, std=self.std)
        else:
            train_set = TSDataset(X=X)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        # initialize the model
        self.model = TranADModel(batch=self.batch_size, feats=self.feats, slidingWindow=self.slidingWindow)

        # move to device and print model information
        self.model = self.model.to(self.device)
        if self.verbose:
            print(self.model)

        # train the autoencoder to find the best one
        self._train_TranAD(train_loader)

        self.model.load_state_dict(self.best_model_dict)
        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self

    def _train_TranAD(self, train_loader):
        """Internal function to train the TranAD model

        Parameters
        ----------
        train_loader : torch dataloader
            Train data.
        """
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)

        self.best_loss = float('inf')
        self.best_model_dict = None

        for epoch in range(1, self.epochs+1):
            overall_loss = []
            for x, data_idx in train_loader:
                x = x.to(self.device).float()
                x = x.unsqueeze(-1)
                bs = x.shape[0]
                x = x.permute(1, 0, 2)
                elem = x[-1, :, :].view(1, bs, self.feats)

                self.optimizer.zero_grad()
                z = self.model(x, elem)
                loss = (1 / epoch) * self.loss_fn(z[0], elem) + (1 - 1 / epoch)* self.loss_fn(z[1], elem)
                loss.backward(retain_graph=True)

                self.optimizer.step()
                overall_loss.append(loss.item())
            self.scheduler.step()
            if self.verbose:
                print('epoch {epoch}: training loss {train_loss} '.format(
                    epoch=epoch, train_loss=np.mean(overall_loss)))

            # track the best model so far
            if np.mean(overall_loss) <= self.best_loss:
                # print("epoch {ep} is the current best; loss={loss}".format(ep=epoch, loss=np.mean(overall_loss)))
                self.best_loss = np.mean(overall_loss)
                self.best_model_dict = self.model.state_dict()

    def decision_function(self, X, measure=None):
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
        check_is_fitted(self, ['model', 'best_model_dict'])

        n_samples, n_features = X.shape
        if n_features == 1: 
            # Converting time series data into matrix format
            X = Window(window = self.slidingWindow).convert(X).to_numpy()

        X = check_array(X)
        X = MinMaxScaler(feature_range=(0,1)).fit_transform(X.T).T

        # note the shuffle may be true but should be False
        if self.preprocessing:
            dataset = TSDataset(X=X, mean=self.mean, std=self.std)
        else:
            dataset = TSDataset(X=X)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False)
        # enable the evaluation mode
        self.model.eval()
        outlier_scores = []
        
        with torch.no_grad():
            for x, data_idx in dataloader:
                x = x.to(self.device).float()

                bs = x.shape[0]
                x = x.unsqueeze(-1)
                x = x.permute(1, 0, 2)
                elem = x[-1, :, :].view(1, bs, 1)
                _, z = self.model(x, elem)
                loss = F.mse_loss(z, elem, reduction="none")[0].squeeze(1)
                outlier_scores.append(loss.cpu())

        outlier_scores = torch.cat(outlier_scores, dim=0)
        outlier_scores = outlier_scores.numpy()

        # padded decision scores
        if outlier_scores.shape[0] < n_samples:
            outlier_scores = np.array([outlier_scores[0]]*math.ceil((self.slidingWindow-1)/2) + 
                        list(outlier_scores) + [outlier_scores[-1]]*((self.slidingWindow-1)//2))
        
        return outlier_scores