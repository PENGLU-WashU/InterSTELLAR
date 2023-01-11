# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
from InterSTELLAR_Utility import train_InterSTELLAR, validate_InterSTELLAR
from InterSTELLAR_Network import InterSTELLAR

"""Train InterSTELLAR"""
def train(datasets, in_channels, out_channels, epoch_num = 200, n_classes = 3, learning_rate = 1e-3, k_sample_val = 8, 
          eta = 0.5, batch_size = 8, reg_weight = 3e-5, results_dir = None, device = None):
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    print('Done!')
    
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('Init loss function...', end=' ')
    loss_fn = nn.CrossEntropyLoss()
    if device.type == 'cuda':
        loss_fn = loss_fn.cuda()
    print('Done!')
    
    print('Init Model...', end=' ')
    model = InterSTELLAR(in_channels = in_channels, out_channels = out_channels, k_sample = k_sample_val,
                         n_classes = n_classes, device = device)
    model = model.to(device)
    model.relocate()
    print('Done!')

    print('Init optimizer ...', end=' ')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    print('Done!')
    
    def get_sample_frequency():
        list_summary = []
        for _ in range(n_classes):
            list_summary.append([])
        for ii in range(len(train_split)):
            list_summary[int(train_split[ii][3])].append(ii)
        custom_distribution = []
        for ii in range(n_classes):
            custom_distribution.append(1/len(list_summary[ii]))
        dist_sum = sum(custom_distribution)
        for ii in range(n_classes):   
            custom_distribution[ii] = custom_distribution[ii]/dist_sum
        sampler_fre = []
        for ii in range(len(train_split)):
            if int(train_split[ii][3]) == 0:
                sampler_fre.append(custom_distribution[0])
            elif int(train_split[ii][3]) == 1:
                sampler_fre.append(custom_distribution[1])
            else:
                sampler_fre.append(custom_distribution[2])
        return sampler_fre
    
    print('Init Loaders...', end=' ')
    custom_sampler = WeightedRandomSampler(weights = get_sample_frequency(), 
                                           num_samples = int(1.3*len(train_split)), replacement = True)
    train_loader =  DataLoader(train_split, batch_size=1, sampler=custom_sampler)
#     train_loader = CustomInput(train_split, n_classes)
    val_loader = DataLoader(val_split, batch_size=1, shuffle=False)
    print('Done!')
    
    print('Training started...')
    val_loss_min = 1e+5
    for epoch in range(epoch_num):
        train_InterSTELLAR(epoch, model, train_loader, optimizer, n_classes, eta, batch_size, reg_weight, loss_fn, device)
        val_loss = validate_InterSTELLAR(epoch, model, val_loader, n_classes, loss_fn, device)
        
        scheduler.step()
        if val_loss_min > val_loss:
            if results_dir:
                val_loss_min = val_loss
                torch.save(model.state_dict(), os.path.join(results_dir, 'saved_model.pt'))
                print('save model')
    print('Training completed!')
    return model


