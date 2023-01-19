# -*- coding: utf-8 -*-
import numpy as np
import torch
from scipy.spatial.distance import squareform, pdist
from scipy.sparse import coo_matrix

def prepare_graphs(raw_graph_data,neighour_thresh,eps_,mean_,std_):
    all_graphs = []
    for m in range(len(raw_graph_data)):    
        cur_features = raw_graph_data[m][0]
        cur_locations = raw_graph_data[m][1]
        cur_labels = raw_graph_data[m][2]
        
        cur_features_log = np.log(cur_features + eps_)
        cur_features_norm = np.divide(cur_features_log - mean_, std_)
        
        Adj_M = squareform(pdist(cur_locations))
        def getDAdj():
            DAdj = (Adj_M < neighour_thresh)*1
            edge_weight = np.exp(-np.square(Adj_M)/(neighour_thresh**2/np.log(neighour_thresh)))
            edge_weight = edge_weight*DAdj
            return DAdj, edge_weight
        
        kNN_adj, kNN_weight = getDAdj()
        edge_weight = []
        kNN_adj = np.array(coo_matrix(kNN_adj).nonzero())
        for ii in range(np.shape(kNN_adj)[1]):
            edge_weight.append(kNN_weight[kNN_adj[0,ii],kNN_adj[1,ii]])
            
        single_graph = []
        edge_index = np.array(kNN_adj)
        edge_attr = np.array(edge_weight)
        single_graph.append(cur_features_norm)
        single_graph.append(edge_index)
        single_graph.append(edge_attr)
        single_graph.append(cur_labels[0])
        
        all_graphs.append(single_graph)   
        print('The {}-th graph has been constructed.'.format(m+1))
    return all_graphs

def prepare_constructed_graphs(raw_graph_data):
    print('Graph construction started...')
    all_graphs = []
    for m in range(len(raw_graph_data)):    
        cur_features = raw_graph_data[m][0]
        cur_edge_index = raw_graph_data[m][1]
        cur_edge_attr = raw_graph_data[m][2]
        cur_labels = raw_graph_data[m][3]
        
        single_graph = []
        single_graph.append(torch.tensor(cur_features, dtype = torch.float))
        single_graph.append(torch.tensor(cur_edge_index, dtype=torch.long).contiguous())
        single_graph.append(torch.tensor(cur_edge_attr, dtype=torch.float))
        single_graph.append(cur_labels.astype(np.int64))
        
        all_graphs.append(single_graph)   
    print('Graph construction completed!')
    return all_graphs
