# Author: Yuhao Kang <yuhaok@uchicago.edu>

import numpy as np
import networkx as nx 
import math
from .sbd_distance import _sbd


#   SBD distance matrix
def construct_sbd_matrix(X_train, y_train):
    n_indexed = len(y_train)
    X_train_distance = np.zeros([n_indexed,n_indexed])
    
    for i in range(n_indexed):
        for j in range(i+1,n_indexed):
            X_train_distance[i][j] = max(_sbd(X_train[i],X_train[j])[0], 0)
            X_train_distance[j][i] = X_train_distance[i][j]
    
    np.fill_diagonal(X_train_distance, 1e4)

    closest = np.argmin(X_train_distance, axis=0)
    y_pred = y_train[closest]

    n_label = len(np.unique(y_train))

    confusionMatrix = np.zeros((n_label,n_label))
    
            
    for i in range(n_label):    # real label is i+1
        for j in range(n_label):  # predicted label is j+1
            tmp1 = y_train==(i+1)
            tmp2 = y_pred==(j+1)
            confusionMatrix[i][j] = np.sum(tmp1*tmp2)/np.sum(tmp1)
            
    adjacencyMatrix = (confusionMatrix + confusionMatrix.T)/2 + 1e-3
    # np.fill_diagonal(adjacencyMatrix,0)
    return adjacencyMatrix


#   clustering based on max-spanning-tree
def clustering_mst(adjacencyMatrix):
    G = nx.from_numpy_matrix(adjacencyMatrix, parallel_edges=False)
    T = nx.maximum_spanning_tree(G,weight='weight')
        
    n_label = len(adjacencyMatrix)
    labelList = [-1]*n_label
    labelList[0] = 0
    
    # BFS
    visited = [0]
    queue = [0]
    while queue:
        s = queue.pop(0)
        for neighbor in T.neighbors(s):
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
                labelList[neighbor] = 1 - labelList[s]
                       
    if len(np.unique(labelList))!=2:
        print('Class Separation Error')
        return
        
    labelList = np.array(labelList)
    return labelList


#   construct anomaly based on 2 clusters
def construct_anomaly(X_train, y_train, clusterIndex, K=3, anomalyRatio=0.05, seed=0, return_statistics=False):
    np.random.seed(seed)
    
    # K is num_anomaly, get num of normal
    N_normal_seg = int(K/anomalyRatio)
    
    ano_labels = np.where(clusterIndex==1)[0]+1  # list of ano labels, add 1 since label index start from 1
    nor_labels = np.where(clusterIndex==0)[0]+1  # list of normal labels
    
    # ano_index in y_train
    ano_index = [i for i in range(len(y_train)) if y_train[i] in ano_labels]
    K = min(K, len(ano_index))
    # selected K ano in y_train
    ano_index_selected = np.random.choice(ano_index, size=K, replace=False)
    
    # corresponding K real label 
    ano_label_selected = y_train[ano_index_selected]
    
    # most frequent selected label
    ano_label_mostfreq = max(ano_label_selected.tolist(), key = ano_label_selected.tolist().count)
    freq_ano_label_mostfreq = sum(ano_label_selected==ano_label_mostfreq)
    
    # min freq for each normal label
    freq_nor_label_min = 20 * freq_ano_label_mostfreq  
    # number of normal label will be used
    num_nor_label = min(math.ceil(N_normal_seg/freq_nor_label_min), len(nor_labels))  
    # real freq for each normal label, here the cardinality of each normal label is the same
    freq_nor_label = math.ceil(N_normal_seg/num_nor_label) 
    
    # selected normal labels
    nor_labels_selected = np.random.choice(nor_labels, size=num_nor_label, replace=False)
    # selected index in y_train
    nor_index_selected = []
    for item in nor_labels_selected:
        tmp = np.where(y_train==item)[0]
        nor_index_selected_item = np.random.choice(tmp, size=freq_nor_label, replace=True)
        nor_index_selected.append(nor_index_selected_item)
    nor_index_selected = np.array(nor_index_selected).ravel()
    
    index_combined = np.concatenate((ano_index_selected, nor_index_selected))
    y_new = np.array([1]*len(ano_index_selected) + [0]*len(nor_index_selected))
    
    order = np.arange(len(index_combined))
    np.random.shuffle(order)
    
    X_new = X_train[index_combined[order]].ravel()
    y_new = y_new[order]
    y_new = np.repeat(y_new, len(X_train[0]))
    
    
    if return_statistics:
        # RC 
        X_nor = X_train[nor_index_selected]
        X_ano = X_train[ano_index_selected]
        X_tot = np.concatenate((X_nor, X_ano), axis=0)
        
        n = len(X_tot)
        sbd_matrix = np.zeros([n,n])
        for i in range(n):
            for j in range(i+1,n):
                sbd_matrix[i,j] = max(_sbd(X_tot[i], X_tot[j])[0],0)
                sbd_matrix[j,i] = sbd_matrix[i,j]
        np.fill_diagonal(sbd_matrix, 5)
        
        min_ = np.min(sbd_matrix, 0)
        mean_ = (np.sum(sbd_matrix,0)-5)/(len(sbd_matrix)-1)
        cr = np.mean(mean_)/np.mean(min_)
    
        # NC
        nor_dist_array = sbd_matrix[:len(X_nor), :len(X_nor)].ravel()
        nor_dist_array = nor_dist_array[nor_dist_array<2]
        avg_nor = np.mean(nor_dist_array)
        
        ano_dist_array = sbd_matrix[len(X_nor):n, len(X_nor):n].ravel()
        ano_dist_array = ano_dist_array[ano_dist_array<2]
        avg_ano = np.mean(ano_dist_array)
        
        nc = np.sqrt(avg_nor/avg_ano)
        
        # NA
        if num_nor_label==1:
            na = 1
        else:
            # split index according to label
            cluster_index_basedon_label_ano = []
            for i in np.unique(ano_label_selected):
                cluster_index_basedon_label_ano.append(ano_index_selected[np.where(ano_label_selected==i)])
            
            cluster_index_basedon_label_nor = []
            for i in np.unique(nor_labels_selected):
                cluster_index_basedon_label_nor.append(nor_index_selected[np.where(nor_labels_selected==i)])
                
                
            # get centroids for nor and ano
            centroids_ano = []
            for i in cluster_index_basedon_label_ano:
                centroids_ano.append(np.mean(X_train[i], axis=0))
                
            centroids_nor = []
            for i in cluster_index_basedon_label_nor:
                centroids_nor.append(np.mean(X_train[i], axis=0))
                
            # dist between centroids of nor and ano
            dist_nor_ano = []
            for i in centroids_ano:
                for j in centroids_nor:
                    dist_nor_ano.append(max(_sbd(i,j)[0],0))
                
            # dist between centroids of nor and nor
            dist_nor_nor = []
            for i in range(len(centroids_nor)):
                for j in range(i+1,len(centroids_nor)):
                    dist_nor_nor.append(max(_sbd(centroids_nor[i],centroids_nor[j])[0],0))
                    
            na = np.min(dist_nor_ano)/np.mean(dist_nor_nor)
        
        return X_new, y_new, len(X_train[0]), round(cr,2), round(nc,2), round(na,2)
    
    return X_new, y_new

