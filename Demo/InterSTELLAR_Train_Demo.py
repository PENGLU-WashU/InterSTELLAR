# -*- coding: utf-8 -*-
import numpy as np
import torch
import argparse
from InterSTELLAR.Graph_Construction_Fuctions import prepare_constructed_graphs
from InterSTELLAR.InterSTELLAR_Train import train

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--dataset", help = "dataset used to build graphs. npy format, along with its folder",
                    default = 'constructed_graph_data.npy', type = str)
parser.add_argument("--myeps", help = "the value used in log transform of the cell data", 
                    default = 1e-4, type = float)
parser.add_argument("--fold_order", help = "10-fold cross validation, specify the fold order, from 1 to 10", 
                    default = 5, type = int)
parser.add_argument("--k_sample_val", help = "k top highest and lowest samples in cell-scale training", 
                    default = 8, type = int)
parser.add_argument("--epoch_num", help = "epoch number", default = 30, type = int)
parser.add_argument("--lr", help = "learning rate", default = 3e-4, type = float)
parser.add_argument("--eta", help = "eta in the loss function, between 0 and 1", default = .85, type = float)
parser.add_argument("--n_classes", help = "number of tissue classes", default = 3, type = int)
parser.add_argument("--out_channels", help = "output channel number", default = 10, type = int)
parser.add_argument("--batch_size", help = "batch size", default = 8, type = int)
parser.add_argument("--lambda_reg", help = "l1 regularization parameter for network weights", 
                    default = 3e-5, type = float)
parser.add_argument("--results_dir", help = "the folder to save the trained weights", default = None, type = str)
parser.add_argument("--GPU", help = "use GPU to train?", default = True, type = str2bool)

args = parser.parse_args()
print(args)

"""0. Check the parameters before implementation"""
if args.fold_order > 10 or args.fold_order < 1:
    print('fold_order must be between 1 and 10!')
    quit()
if args.eta > 1 or args.eta < 0:
    print('fold_order must be between 0 and 1!')
    quit()
if args.GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
if not args.results_dir:
    print('results_dir is not specified, the trained weights will not be saved!')    

"""
1. Graph Construction
"""
graph_data = np.load(args.dataset, allow_pickle = True)
all_graphs = prepare_constructed_graphs(graph_data)
del graph_data

"""
2. Train InterSTELLAR
Here we applied a stratified sampling strategy for training and validation due to label imbalance issue.
"""    
healthy_list = []
tnbc_list = []
non_tnbc_list = []
all_graph_healthy = []
all_graph_tnbc = []
all_graph_non_tnbc = []
datasets = []

for kk in range(len(all_graphs)):
    if int(all_graphs[kk].y) == 0:
        healthy_list.append(kk)
        all_graph_healthy.append(all_graphs[kk])   
    if int(all_graphs[kk].y) == 1:
        tnbc_list.append(kk)
        all_graph_tnbc.append(all_graphs[kk])
    if int(all_graphs[kk].y) == 2:
        non_tnbc_list.append(kk)
        all_graph_non_tnbc.append(all_graphs[kk])
del all_graphs
        
step = 3
test_list_healthy = list(range(7,7+17*step,step))
test_list_tnbc = list(range(4,4+9*step,step))
test_list_non_tnbc = list(range(19,19+47*step,step))

train_val_list_healthy = list(set([*range(0, len(healthy_list))]) - set(test_list_healthy))
train_val_list_tnbc = list(set([*range(0, len(tnbc_list))]) - set(test_list_tnbc))
train_val_list_non_tnbc = list(set([*range(0, len(non_tnbc_list))]) - set(test_list_non_tnbc))

val_list_healthy_idx = list(range(0,len(train_val_list_healthy),7))
val_list_healthy_idx.append(len(train_val_list_healthy))
val_list_tnbc_idx = list(range(0,len(train_val_list_tnbc),4))
val_list_tnbc_idx.append(len(train_val_list_tnbc))
val_list_non_tnbc_idx = list(range(0,len(train_val_list_non_tnbc),19))
val_list_non_tnbc_idx.append(len(train_val_list_non_tnbc))

for kk in range(10):
    print('current fold: ', kk+1)
    val_range_healthy = train_val_list_healthy[val_list_healthy_idx[kk]:val_list_healthy_idx[kk+1]]
    val_range_tnbc = train_val_list_tnbc[val_list_tnbc_idx[kk]:val_list_tnbc_idx[kk+1]]
    val_range_non_tnbc = train_val_list_non_tnbc[val_list_non_tnbc_idx[kk]:val_list_non_tnbc_idx[kk+1]]

    train_range_healthy = list(set(train_val_list_healthy) - set(val_range_healthy))
    train_range_tnbc = list(set(train_val_list_tnbc) - set(val_range_tnbc))
    train_range_non_tnbc = list(set(train_val_list_non_tnbc) - set(val_range_non_tnbc))
    
    train_holder = []
    for ii in train_range_healthy:
        train_holder.append([all_graph_healthy[ii].x, all_graph_healthy[ii].edge_index, 
                     all_graph_healthy[ii].edge_attr, int(all_graph_healthy[ii].y)])
    for ii in train_range_tnbc:
        train_holder.append([all_graph_tnbc[ii].x, all_graph_tnbc[ii].edge_index, 
                     all_graph_tnbc[ii].edge_attr, int(all_graph_tnbc[ii].y)])
    for ii in train_range_non_tnbc:
        train_holder.append([all_graph_non_tnbc[ii].x, all_graph_non_tnbc[ii].edge_index, 
                     all_graph_non_tnbc[ii].edge_attr, int(all_graph_non_tnbc[ii].y)])
        
    val_holder = []
    for ii in val_range_healthy:
        val_holder.append([all_graph_healthy[ii].x, all_graph_healthy[ii].edge_index, 
                     all_graph_healthy[ii].edge_attr, int(all_graph_healthy[ii].y)])
    for ii in val_range_tnbc:
        val_holder.append([all_graph_tnbc[ii].x, all_graph_tnbc[ii].edge_index, 
                     all_graph_tnbc[ii].edge_attr, int(all_graph_tnbc[ii].y)])
    for ii in val_range_non_tnbc:
        val_holder.append([all_graph_non_tnbc[ii].x, all_graph_non_tnbc[ii].edge_index, 
                     all_graph_non_tnbc[ii].edge_attr, int(all_graph_non_tnbc[ii].y)])
        
    datasets.append((train_holder, val_holder))

del all_graph_healthy, all_graph_tnbc, all_graph_non_tnbc

print('The model is training on the {}-th fold.'.format(args.fold_order))
model = train(datasets[args.fold_order-1], np.shape(train_holder[0][0])[1], args.out_channels, 
              args.epoch_num, args.n_classes, args.lr, args.k_sample_val, args.eta, args.batch_size, 
              args.lambda_reg, args.results_dir, device)