# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:42:28 2024

@author: bugatap
"""

import os

import pandas as pd
import numpy as np

from helper_code import find_records, load_dx

import torch
from pytorch_model_summary import summary

from dataset import FileImageDataset
from networks import ConvnextNetwork
from net_utils_sup import NetworkModel
from losses import WeightedBCEWithLogitsLoss
from metrics import MultilabelF1Macro
from convnext_config import Config

# get metadata
def get_metadata(source_folder):
    # find records
    records = find_records(source_folder)
    num_records = len(records)

    if num_records == 0:
        print('No data was provided.')
        exit(-1)

    # Extract the features and labels.
    print('Extracting features and labels from the data...')
    
    label_set = ['CD','HYP','MI','NORM','STTC']
    
    output = []

    for i in range(num_records):
        record_id = records[i]
        record = os.path.join(source_folder, record_id)

        # Extract the features from the image, but only if the image has one or more dx classes.
        dx = load_dx(record)
                
        #ecg_id = records[i].split('/')[-1]
        output_dict = {'RECORD_ID': record_id}
        for label in label_set:
            if label in dx:
                output_dict[label] = 1
            else:
                output_dict[label] = 0
        
        output.append(output_dict)
    
    mddf = pd.DataFrame(output)
    return mddf

# count trainable params
def count_params(network):
    return sum(p.numel() for p in network.parameters() if p.requires_grad)

# split weights into two smaller files
def save_to_two_files(state_dict, output_path, f_prefix):
    state_dict_A, state_dict_B = {}, {}
    for i, (key, val) in enumerate(state_dict.items()):
        if i < len(state_dict) // 2:
            state_dict_A[key] = val
        else:
            state_dict_B[key] = val
    torch.save(state_dict_A, output_path + '/' + f_prefix + '_A.h5')
    torch.save(state_dict_B, output_path + '/' + f_prefix + '_B.h5')

# merge files into one state dict    
def load_from_two_files(path_to_load, f_prefix):
    state_dict_A = torch.load(path_to_load + '/' + f_prefix + '_A.h5')
    state_dict_B = torch.load(path_to_load + '/' + f_prefix + '_B.h5') 
    return {**state_dict_A, **state_dict_B}

# training networks - only classifiaction layer
def train(data_folder, pretrained_path, output_path):
    
    # labels
    labels = ['NORM','CD','HYP','MI','STTC']
    
    # configuration
    cfg = Config()
    print(cfg)
    
    # mddf
    mddf = get_metadata(data_folder)
    #mddf = mddf[:1000]
    
    # networks are pretrained from 5-fold cross validation 
    for i in range(5):
        
        # ECG IDs
        train_ids = mddf.RECORD_ID.values

        # target
        train_target = mddf.loc[:, labels].values
        
        # datasets and loaders
        train_ds = FileImageDataset(data_folder, train_ids, train_target, random_erasing=cfg.random_erasing)
        
        train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=cfg.batch_size, num_workers=cfg.P)
    
        # make convnext model
        network = ConvnextNetwork(n_outputs=5)
        
        # freeze ConvNext
        print()
        print('Fold:', i)
        print('Trainable params before freezing: ',  count_params(network))
        for param in network.convnext.parameters():
            param.requires_grad = False
        for param in network.convnext.classifier[2].parameters():   
            param.requires_grad = True 
        print('ConvNext weights frozen. Trainable params: ',  count_params(network))
        
        # summary
        if i == 0:
            summary(network, torch.rand(1, 3, 242, 300), show_hierarchical=True, 
                    print_summary=True, show_parent_layers=True, max_depth=None)
        
        # to device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        network = network.to(device=device)
        
        # optimizer
        optimizer = torch.optim.AdamW(params=filter(lambda p:p.requires_grad, network.parameters()), lr=0.001, 
                                      weight_decay=cfg.WD, betas=(0.9, 0.99)) 
    
        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.max_lr, 
                                                        pct_start=cfg.pct_start, anneal_strategy=cfg.anneal_strategy,
                                                        epochs=cfg.n_epochs, steps_per_epoch=len(train_dl))
            
        # loss function
        loss_fn = WeightedBCEWithLogitsLoss(pos_weight=torch.Tensor(np.array(cfg.pos_weight)).to(device=device), 
                                            neg_weight=torch.Tensor(np.array(cfg.neg_weight)).to(device=device), 
                                            reduction='mean', label_smoothing=cfg.label_smoothing)

            
        # eval. metrics
        eval_metrics = [MultilabelF1Macro()]
        
        
        # construct model
        model = NetworkModel(network=network, optimizer=optimizer, scheduler=scheduler, 
                 loss_function=loss_fn, metrics=eval_metrics, use_metric='f1', accum_iters=1,
                 n_repeats=1, pred_agg=None,
                 verbose=True, grad_norm=None, use_16fp=cfg.use_16fp, 
                 freeze_BN=False, output_fn=torch.sigmoid, mixup=cfg.mixup, 
                 rank=None, world_size=1)

        # load weights from pretrained
        state_dict = load_from_two_files(pretrained_path, f_prefix='weights{}'.format(i))
        model.network.load_state_dict(state_dict)
        #model.load_weights(pretrained_path + '/' + pretrained_file)
        
        # fit
        model_file = 'weights{}.h5'.format(i)
        model_path = output_path + '/' + model_file
        model.fit(loader=train_dl, loader_valid=None, n_epochs=cfg.n_epochs, model_file=model_path)

        # save weights
        model.save_weights(model_path)
        
   