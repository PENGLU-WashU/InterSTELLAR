# -*- coding: utf-8 -*-
import numpy as np
import torch

"""
Adopted from https://github.com/mahmoodlab/CLAM
Calcuate training and validation accuracy.
"""
class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

"""
Training code
"""
def train_InterSTELLAR(epoch, model, loader, optimizer, n_classes, eta, batch_size, reg_weight, loss_fn = None, device = None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    cell_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_cell_scale_loss_all = 0.
    cell_count = 0

    batch_num = 0
    loss_summary = 0
    for batch_idx, (x, edge_index, edge_attr, y) in enumerate(loader):
        x, edge_index, edge_attr = x[0].to(device), edge_index[0].to(device), edge_attr[0].to(device)
        label =  y.to(device)
        logits, Y_prob, Y_hat, _, _, total_cell_scale_loss, cell_preds, cell_targets = model(x, edge_index, edge_attr, label=label)
        
        cell_logger.log_batch(cell_preds, cell_targets)
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)

        cell_count+=1
        train_cell_scale_loss_all += total_cell_scale_loss.item()
        
        total_loss = eta * loss + (1-eta) * total_cell_scale_loss 
        train_loss += loss.item()
        loss_summary += total_loss
        
        l1_reg = None
        for W in model.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
        loss_summary += reg_weight*l1_reg
        
        batch_num += 1
        if batch_num == batch_size or batch_idx == len(loader)-1:
            batch_num = 0
            loss_summary.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_summary = 0
    
    train_loss /= len(loader)
    train_cell_scale_loss_all /= cell_count
    print('\nEpoch: {}:\nTraining set:\n  tissue_scale_train_loss: {:.4f}, cell_scale_train_loss:  {:.4f}'.format(epoch, train_loss, train_cell_scale_loss_all))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('  tissue-scale class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
    
    for i in range(2):
        acc, correct, count = cell_logger.get_summary(i)
        print('  cell-scale class {} prediction accuracy {}: correct {}/{}'.format(i, acc, correct, count))

"""
Validation code
"""
def validate_InterSTELLAR(epoch, model, loader, n_classes, loss_fn = None, device = None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    cell_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_cell_scale_loss_all = 0.
    cell_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    with torch.no_grad():
        for batch_idx, (x, edge_index, edge_attr, y) in enumerate(loader):
            x, edge_index, edge_attr = x[0].to(device), edge_index[0].to(device), edge_attr[0].to(device)
            label =  y.to(device)
            logits, Y_prob, Y_hat, _, _, total_cell_scale_loss, cell_preds, cell_targets = model(x, edge_index, edge_attr, label=label)
            
            cell_logger.log_batch(cell_preds, cell_targets)
            acc_logger.log(Y_hat, label)
            loss = loss_fn(logits, label)
            val_loss += loss.item()
            
            cell_count+=1
            val_cell_scale_loss_all += total_cell_scale_loss.item()

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

    val_loss /= len(loader)
    val_cell_scale_loss_all /= cell_count

    print('\nValidation set:\n  tissue_scale_val_loss: {:.4f}, cell_scale_val_loss: {:.4f}'.format(val_loss, val_cell_scale_loss_all))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('  class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
    
    for i in range(2):
        acc, correct, count = cell_logger.get_summary(i)
        print('  class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    return val_loss