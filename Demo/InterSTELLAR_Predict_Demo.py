# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from InterSTELLAR.Graph_Construction_Fuctions import prepare_constructed_graphs
from InterSTELLAR.InterSTELLAR_Network import InterSTELLAR
from sklearn.metrics import confusion_matrix, f1_score
import scipy.io as sio

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
parser.add_argument("--trained_weights", help = "load a trained weights file to model", type = str)
parser.add_argument("--dataset", help = "dataset used to build graphs. npy format, along with its folder",
                    default = 'constructed_graph_data.npy', type = str)
parser.add_argument("--n_classes", help = "number of tissue classes", default = 3, type = int)
parser.add_argument("--out_channels", help = "output channel number", default = 10, type = int)
parser.add_argument("--GPU", help = "use GPU to train?", default = True, type = str2bool)
parser.add_argument("--save_results", help = "save the predicted tissue and cell-scale results?", default = False, type = str2bool)

args = parser.parse_args()
print(args)

"""0. Check the parameters before implementation"""
if args.GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

"""
1. Graph Construction
"""
graph_data = np.load(args.dataset, allow_pickle = True)
all_graphs = prepare_constructed_graphs(graph_data)

"""
2. Build Test Set
Here we applied a stratified sampling strategy for training and validation due to label imbalance issue.
"""    
all_graph_healthy = []
all_graph_tnbc = []
all_graph_non_tnbc = []

for kk in range(len(all_graphs)):
    if int(all_graphs[kk][3]) == 0:
        all_graph_healthy.append(all_graphs[kk])   
    if int(all_graphs[kk][3]) == 1:
        all_graph_tnbc.append(all_graphs[kk])
    if int(all_graphs[kk][3]) == 2:
        all_graph_non_tnbc.append(all_graphs[kk])
del all_graphs
        
step = 3
test_list_healthy = list(range(7,7+17*step,step))
test_list_tnbc = list(range(4,4+9*step,step))
test_list_non_tnbc = list(range(19,19+47*step,step))

test_holder = []
for ii in range(len(test_list_healthy)):
    cur_idx = test_list_healthy[ii]
    test_holder.append(all_graph_healthy[cur_idx])
for ii in range(len(test_list_tnbc)):
    cur_idx = test_list_tnbc[ii]
    test_holder.append(all_graph_tnbc[cur_idx])
for ii in range(len(test_list_non_tnbc)):
    cur_idx = test_list_non_tnbc[ii]
    test_holder.append(all_graph_non_tnbc[cur_idx])
del all_graph_healthy, all_graph_tnbc, all_graph_non_tnbc
loader = DataLoader(test_holder, batch_size=1, shuffle=False)
"""
3. Load model and predict
"""
ckpt = torch.load(args.trained_weights)
model = InterSTELLAR(in_channels = np.shape(test_holder[0][0])[1], out_channels = args.out_channels, 
                        n_classes = args.n_classes, device = device)
model.load_state_dict(ckpt, strict=True)
model = model.to(device)

all_predictions = []
all_labels = []
all_attention_maps = []
for _, (x, edge_index, edge_attr, y) in enumerate(loader):
    x, edge_index, edge_attr = x[0].to(device), edge_index[0].to(device), edge_attr[0].to(device)
    label =  y.to(device)
    with torch.no_grad():
        _, Y_prob, _, Attention_score, _, _, _, _ = model(x, edge_index, edge_attr, label)
        
    all_predictions.append(np.argmax(Y_prob.cpu().numpy(), axis = 1))
    all_labels.append(label.item())
    all_attention_maps.append(Attention_score.cpu().numpy())

""" Evaluate accuracy """
all_labels = np.array(all_labels).astype(int)
all_predictions = np.array(all_predictions).astype(int)
con_mat = np.array(confusion_matrix(all_labels, all_predictions))
balanced_accuracy = np.zeros((1,3))
for ii in range(3):
    balanced_accuracy[0,ii] = con_mat[ii,ii]/np.sum(con_mat[ii,:], axis  = -1) 
balanced_accuracy = np.mean(balanced_accuracy)
print('Balanced accuracy: ' + str(balanced_accuracy))    
print('F1 score: ' + str(f1_score(all_labels, all_predictions, average = 'macro')))
"""
4. Save results
"""
if args.save_results:
    mdict = {"true_label": all_labels, "predicted_label": all_predictions}
    sio.savemat(r'true_predicted_labels.mat', mdict)
    mdict = {"predicted_attention_map" + str(ii): all_attention_maps[ii] for ii in range(len(all_attention_maps))}
    sio.savemat(r'predicted_attention_maps.mat', mdict)
    