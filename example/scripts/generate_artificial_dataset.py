import numpy as np
import pandas as pd
from TSB_UAD.transformer.artificialConstruction import construct_sbd_matrix,clustering_mst,construct_anomaly
import argparse


def main(name, K, anomalyRatio, seed):
    
    data_path = '../../data/UCR2018-NEW/'+name+'/'+name+'_TRAIN'
    data_train = np.loadtxt(data_path, delimiter=',')
    X_train_raw = data_train[:,1:]
    y_train_raw = data_train[:,0].astype(int)
            
    
    adjacencyMatrix = construct_sbd_matrix(X_train_raw, y_train_raw)
    clusterIndex = clustering_mst(adjacencyMatrix)
    
    
    synX, synY = construct_anomaly(X_train_raw, y_train_raw, 
                      clusterIndex, K, anomalyRatio, seed)
    
    
    data = synX.astype(float)
    label = synY
    
    df_data = pd.DataFrame(data)
    df_label = pd.DataFrame(label)
    
    df = pd.concat([df_data, df_label], axis=1)
    
    data_name = name+'_'+str(K)+'_'+str(anomalyRatio)+'_'+str(seed)
    df.to_csv('../data/artificial/'+data_name+'.out', header=False, index=False)
    
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--K', type=int, required=True)
parser.add_argument('--anomalyRatio', type=float, required=True)
parser.add_argument('--seed', type=int, required=True)

# Parse the argument
args = parser.parse_args()

# python generate_artificial_dataset.py --name Symbols --K 2 --anomalyRatio 0.02 --seed 5

if __name__ == "__main__":
    main(args.name, args.K, args.anomalyRatio, args.seed)