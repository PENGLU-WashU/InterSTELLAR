# -*- coding: utf-8 -*-
"""
Attention module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, LayerNorm

"""Attentino pooling module"""
class Attention_module(nn.Module):
    def __init__(self, D1 = 20, D2 = 10):
        super(Attention_module, self).__init__()
        self.attention_Tanh = [
            nn.Linear(D1, D2),
            nn.Tanh()]
        
        self.attention_Sigmoid = [
            nn.Linear(D1, D2),
            nn.Sigmoid()]

        self.attention_Tanh = nn.Sequential(*self.attention_Tanh)
        self.attention_Sigmoid = nn.Sequential(*self.attention_Sigmoid)
        self.attention_Concatenate = nn.Linear(D2, 1)

    def forward(self, x): # 20->10->2
        tanh_res = self.attention_Tanh(x)
        sigmoid_res = self.attention_Sigmoid(x)
        Attention_score = tanh_res.mul(sigmoid_res)
        Attention_score = self.attention_Concatenate(Attention_score)  # N x n_classes
        return Attention_score, x

"""Initial weights"""
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
"""InterSTELLAR network structure"""
class InterSTELLAR(nn.Module):
    def __init__(self, in_channels = 30, out_channels = 10, k_sample = 8, n_classes = 3, device = None):
        super(InterSTELLAR, self).__init__()
        self.attention_net = Attention_module(D1 = 2*int(out_channels), D2 = int(out_channels))
        self.classifiers = nn.Linear(2*int(out_channels), n_classes)
        cell_scale_classifiers = [nn.Linear(2*int(out_channels), 2) for _ in range(n_classes)]

        self.cell_scale_classifiers = nn.ModuleList(cell_scale_classifiers)
        self.k_sample = k_sample
        
        from topk.svm import SmoothTop1SVM
        cell_scale_loss_fn = SmoothTop1SVM(n_classes = 2)
        if device.type == 'cuda':
            cell_scale_loss_fn = cell_scale_loss_fn.cuda()
        self.cell_scale_loss_fn = cell_scale_loss_fn
        
        self.n_classes = n_classes
        self.device = device
        
        self.gcn1 = GCNConv(in_channels, 4*out_channels)
        self.bn1 = LayerNorm(4*out_channels)
        self.gcn2 = GCNConv(4*out_channels, 4*out_channels)
        self.bn2 = LayerNorm(4*out_channels)
        self.selu = nn.SELU()
        self.fnn_layer = nn.Linear(4*out_channels, 2*out_channels)
        initialize_weights(self)

    def relocate(self):
        self.attention_net = self.attention_net.to(self.device)
        self.classifiers = self.classifiers.to(self.device)
        self.cell_scale_classifiers = self.cell_scale_classifiers.to(self.device)
    
    """
    cell_scale evaluation for in-the-class attention branch
    adopted from https://github.com/mahmoodlab/CLAM
    """
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_second_positive_targets(length, device):
        return torch.full((length, ), -1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
        
    def cell_scale_eval(self, A, h, classifier): 
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, self.device)
        n_targets = self.create_negative_targets(self.k_sample, self.device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_cells = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_cells)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        cell_scale_loss = self.cell_scale_loss_fn(logits, all_targets)
        return cell_scale_loss, all_preds, all_targets

    def forward(self, x, edge_index, edge_attr, label=None):
        x, edge_index, edge_weight = x, edge_index, edge_attr
        
        x_1 = self.gcn1(x, edge_index, edge_weight=edge_weight)
        x_1 = self.bn1(x_1)
        x_1 = self.selu(x_1)

        x_2 = self.gcn2(x_1, edge_index, edge_weight=edge_weight)
        x_2 = self.bn2(x_2)
        x_2 = self.selu(x_2)

        x_3 = self.fnn_layer(x_2)
        x_3 = self.selu(x_3)
        
        Attention_scores, x_out = self.attention_net(x_3)      
        Attention_scores = torch.transpose(Attention_scores, 1, 0)
        Attention_scores = F.softmax(Attention_scores, dim=1)

        cell_scale_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()
        total_cell_scale_loss = 0.0
        cell_preds = []
        cell_targets = []

        for i in range(len(self.cell_scale_classifiers)):
            cell_scale_label = cell_scale_labels[i].item()
            classifier = self.cell_scale_classifiers[i]
            if cell_scale_label == 1:
                cell_scale_loss, preds, targets = self.cell_scale_eval(Attention_scores, x_out, classifier)
                cell_preds.extend(preds.cpu().numpy())
                cell_targets.extend(targets.cpu().numpy())
            else:
                continue
            total_cell_scale_loss += cell_scale_loss

        x_pooled = torch.mm(Attention_scores, x_out) 
        logits = self.classifiers(x_pooled)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        return logits, Y_prob, Y_hat, Attention_scores, x_out, total_cell_scale_loss, cell_preds, cell_targets