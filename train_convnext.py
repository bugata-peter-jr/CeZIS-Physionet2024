# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:42:28 2024

@author: bugatap
"""

import os

import pandas as pd
import numpy as np
import joblib

from helper_code import find_records, load_labels, load_images

import torch
# from pytorch_model_summary import summary

from dataset import FileImageDataset
from network import ConvnextNetwork, ConvnextNetworkSimple
from net_utils import NetworkModel
from losses import WeightedBCEWithLogitsLoss
from metrics import MultilabelF1Macro
from config import Config
from dataset import resize_image, image_to_tensor


# load rotation models
def load_rotation_models():
    pretrained_folder = './pretrained'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
           
    networks_rot = []
    
    # create and load networks 
    for i in range(5):
        f_prefix = 'weights{:d}'.format(i)
        
        # create network with correct device - rotation
        network = ConvnextNetworkSimple(weights=None, progress=False, n_classes=1, stochastic_depth_prob=0.1)
        network = network.to(device=device)
        
        # to evaluation mode
        network.eval()
        
        try:
            rot_pretrained_path = pretrained_folder + '/rotation'
            state_dict = load_from_two_files(rot_pretrained_path, f_prefix, map_loc=device)
            network.load_state_dict(state_dict, strict=True)
        except:
            print('Weights file: {:s} not available.'.format(rot_pretrained_path + '/' + f_prefix))
            exit(-2)
            
        networks_rot.append(network)

    return networks_rot


# get rotation metadata
def get_rotation(source_folder):

    if os.path.exists('./rotation_dict.pkl'):    
        rotation_dict = joblib.load('./rotation_dict.pkl')
        return rotation_dict
    
    # load rotation models
    networks_rot = load_rotation_models()        
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # find records
    records = find_records(source_folder)
    num_records = len(records)

    if num_records == 0:
        print('No data was provided.')
        exit(-1)

    # Predict rotation.
    print('Predictiong rotation for images...', flush=True)
        
    rotation_dict = {}
    output = []

    for i in range(num_records):
        record_id = records[i]
        record = os.path.join(source_folder, record_id)
    
        # load ecg images
        ecg_images = load_images(record)
        
        for j in range(len(ecg_images)):
            ecg_image = ecg_images[j]

            # load image                      
            ecg_image = resize_image(ecg_image)

            with torch.no_grad():
                # to tensor
                x = image_to_tensor(ecg_image, device=device)                
                x = x.unsqueeze(0)
                #print('x.shape:', x.shape)
                pred_list = [network(x).cpu().numpy()[0,0] for network in networks_rot]
                avg_angle_n = sum(pred_list) / len(pred_list)   
             
            # round to integer
            avg_angle = int(round(30 * avg_angle_n))
                
            rotation_dict[str(record_id) + '-' + str(j)] = avg_angle
            output_dict = {'RECORD_ID': record_id, 'COPY_ID': j, 'AVG_ANGLE': avg_angle}
            output.append(output_dict)
    
    joblib.dump(rotation_dict, './rotation_dict.pkl', protocol=0)
    
    mddf = pd.DataFrame(output)
    mddf.to_csv('./rotation.csv', index=False)
    
    return rotation_dict


# get metadata
def get_metadata(source_folder, labels):
    # find records
    records = find_records(source_folder)
    num_records = len(records)

    if num_records == 0:
        print('No data was provided.')
        exit(-1)

    # Extract the features and labels.
    print('Extracting features and labels from the data...', flush=True)
        
    output = []

    for i in range(num_records):
        record_id = records[i]
        record = os.path.join(source_folder, record_id)

        # Extract the features from the image, but only if the image has one or more dx classes.
        dx = load_labels(record)
        output_dict = {'RECORD_ID': record_id}
        for label in labels:
            if label in dx:
                output_dict[label] = 1
            else:
                output_dict[label] = 0
        
        output.append(output_dict)
    
    mddf = pd.DataFrame(output)
    
    # additional renaming columns 
    mddf.rename(columns={'Acute MI':'Acute_MI', 'Old MI':'Old_MI', 'AFIB/AFL':'AFIB_AFL'}, inplace=True)
    mddf.to_csv('./metadata.csv', index=False)
    return mddf


# count trainable params
def count_params(network):
    return sum(p.numel() for p in network.parameters() if p.requires_grad)

# split weights into two smaller files
# beacuse files larger than 50 MB are not allowed for GitHub
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
# beacuse weights were split to two files
def load_from_two_files(path_to_load, f_prefix, map_loc):
    state_dict_A = torch.load(path_to_load + '/' + f_prefix + '_A.h5', map_location=map_loc)
    state_dict_B = torch.load(path_to_load + '/' + f_prefix + '_B.h5', map_location=map_loc) 
    return {**state_dict_A, **state_dict_B}

# training networks - only classifiaction layer
def train(data_folder, pretrained_path, output_path):
    
    # labels
    labels = ['NORM', 'Acute_MI', 'Old_MI', 'STTC', 'CD', 'HYP', 'PAC', 'PVC', 'AFIB_AFL', 'TACHY', 'BRADY']
    
    # configuration
    cfg = Config()
    print(cfg)
    
    # getting metadata from header files
    mddf = get_metadata(data_folder, cfg.classes)

    # getting rotation
    rotation_dict = get_rotation(data_folder)    
    
    # image folder - same as data folder
    image_folder = data_folder
    
    # networks are pretrained from 5-fold cross validation 
    for i in range(5):
        
        # ECG IDs
        train_ids = mddf.RECORD_ID.values

        # target
        train_target = mddf.loc[:, labels].values
        
        # datasets and loaders
        train_ds = FileImageDataset(image_folder, train_ids, rotation_dict, train_target, random_erasing=cfg.random_erasing)
        
        train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=cfg.batch_size, num_workers=cfg.P)
    
        # make convnext model
        network = ConvnextNetwork(weights=cfg.pretrained, progress=cfg.progress, n_classes=11, stochastic_depth_prob=0.1)

        # freeze ConvNext
        # only classification head will be trained
        print()
        print('Fold:', i)
        print('Trainable params before freezing: ',  count_params(network))
        for param in network.convnext.parameters():
            param.requires_grad = False
        for param in network.convnext.classifier.linear.parameters():   
            param.requires_grad = True 
        print('ConvNext weights frozen. Trainable params: ',  count_params(network))

                
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
                 loss_fn=loss_fn, metric_fn=eval_metrics, use_metric='f1', accum_iters=cfg.accum_iters,
                 n_repeats=1, pred_agg=None,
                 verbose=True, grad_norm=None, use_16fp=cfg.use_16fp, 
                 freeze_BN=False, output_fn=torch.sigmoid, mixup=cfg.mixup, 
                 rank=None, world_size=1)

        # summary
        if i == 0:
            model.summary((1, 3, 512, 600))

        # load weights from pretrained
        state_dict = load_from_two_files(pretrained_path, f_prefix='weights{}'.format(i), map_loc=device)
        model.network.load_state_dict(state_dict)
        
        # fit
        model_file = 'weights{}.h5'.format(i)
        model_path = output_path + '/' + model_file
        model.fit(loader=train_dl, loader_valid=None, n_epochs=cfg.n_epochs, model_file=None)

        # save weights
        model.save_weights(model_path)
        
   