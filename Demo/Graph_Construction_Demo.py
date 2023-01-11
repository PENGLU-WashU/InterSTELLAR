# -*- coding: utf-8 -*-
import argparse
from InterSTELLAR.Graph_Construction_Fuctions import prepare_graphs
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", help = "dataset used to build graphs from raw data. npy format, along with its folder",
                    default = 'graph_data.npy', type = str)
parser.add_argument("--neighbour_thresh", help = "the thresh value used to determine \
                    if any two cells are neighbours", default = 40, type = int)
parser.add_argument("--myeps", help = "the value used in log transform of the cell data", 
                    default = 1e-4, type = float)
parser.add_argument("--built_graph_data", help = "built graphs from raw data. npy format, along with its folder",
                    default = 'constructed_graph_data.npy', type = str)
args = parser.parse_args()
print(args)

graph_data = np.load(args.dataset, allow_pickle = True)
feature_matrix_log_mean,feature_matrix_log_std = graph_data[-2:]
graph_data = graph_data[0:-2]
all_graphs = prepare_graphs(graph_data,args.neighbour_thresh,args.myeps,feature_matrix_log_mean,feature_matrix_log_std)
np.save(args.built_graph_data,all_graphs,allow_pickle=True)
print('Graph saved as {}.'.format(args.built_graph_data))