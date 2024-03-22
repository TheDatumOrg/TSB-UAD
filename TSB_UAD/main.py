import pandas as pd
from utils.slidingWindows import find_length_rank
from model_wrapper import *
import torch
import random

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

data_direc = '../data/benchmark/ECG/MBA_ECG805_data.out'
df = pd.read_csv(data_direc, header=None).dropna().to_numpy()
data = df[:5000,0].astype(float).reshape(-1, 1)
label = df[:5000,1].astype(int)
slidingWindow = find_length_rank(data, rank=1)

anomaly_score = run_iforest_dev(data, periodicity=1, n_estimators=100, n_jobs=1)
anomaly_score = run_lof_dev(data, periodicity=1, n_neighbors=30, n_jobs=1)
anomaly_score = run_poly_dev(data, periodicity=1, power=3, n_jobs=1)
anomaly_score = run_matrix_profile_dev(data, periodicity=1,  n_jobs=1)
anomaly_score = run_pca_dev(data, periodicity=1)
anomaly_score = run_hbos_dev(data, periodicity=1)
anomaly_score = run_ocsvm_dev(data, periodicity=1)
anomaly_score = run_ae_dev(data, periodicity=1, hidden_neurons=[32, 16, 32])
anomaly_score = run_cnn_dev(data, periodicity=1, num_channel=[32, 32, 40], activation='relu')
anomaly_score = run_lstm_dev(data, periodicity=1, hidden_dim=20, activation='relu')
anomaly_score = run_TranAD(data, periodicity=1)

evaluation_result = get_metrics(anomaly_score, label, metric='all', slidingWindow=slidingWindow)

print('evaluation_result: ', evaluation_result)