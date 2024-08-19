#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import numpy as np
import os

from helper_code import load_images

import torch
import pandas as pd

from train_convnext import train, load_from_two_files
from network import ConvnextNetwork, ConvnextNetworkSimple
from config import Config
from dataset import resize_image, rotate_image, image_to_tensor

from imgaug import augmenters as iaa
from PIL import Image


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your digitization model.
def train_models(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training of dx model...')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    pretrained_path = './pretrained/dx'
    train(data_folder, pretrained_path, model_folder)

    return

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_models(model_folder, verbose):
    # no digitization model
    digitization_model = None

    # classification model
    classification_model = Model(eval_mode=False)
    classification_model.load(model_folder)

    return digitization_model, classification_model

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you did not train one of the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):
    # no digitization model
    signals = None
    
    # classification model
    probs = classification_model.predict(record)
    classes = classification_model.classes
    labels = [classes[i] for i in range(len(classes)) if probs[i] >= 0.5]

    return signals, labels

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################
    
class Model(object):
    
    def __init__(self, eval_mode=False):
        # configuration
        cfg = Config()
        
        # weighting of predictions - weight of prediction of pretrained model
        self.w_pretrained = cfg.w_pretrained
        
        # networks pretrained to predict rotation angle
        self.networks_rot = []        
        
        # networks pretrained to predict labels
        self.networks_dx = []
        self.networks_dx_finetuned = []
        
        # evaluation mode
        self.eval_mode = eval_mode
        if self.eval_mode:
            self.ecg_folds = pd.read_csv('./folds.csv')
        
        # device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # classes
        self.classes = cfg.classes
        
    # load model from folder    
    def load(self, folder):
        pretrained_folder = './pretrained'
                
        # create and load networks 
        for i in range(5):
            f_prefix = 'weights{:d}'.format(i)
            
            # create network with correct device - rotation
            network = ConvnextNetworkSimple(weights=None, progress=False, n_classes=1, stochastic_depth_prob=0.1)
            network = network.to(device=self.device)
            
            # to evaluation mode
            network.eval()
            
            try:
                rot_pretrained_path = pretrained_folder + '/rotation'
                state_dict = load_from_two_files(rot_pretrained_path, f_prefix, map_loc=self.device)
                network.load_state_dict(state_dict, strict=True)
            except:
                print('Weights file: {:s} not available.'.format(rot_pretrained_path + '/' + f_prefix))
                exit(-2)
            
            self.networks_rot.append(network)

            # create network with correct device - dx
            network = ConvnextNetwork(weights=None, progress=False, n_classes=11, stochastic_depth_prob=0.1)
            network = network.to(device=self.device)
            
            # to evaluation mode
            network.eval()
            
            try:
                dx_pretrained_path = pretrained_folder + '/dx' 
                state_dict = load_from_two_files(dx_pretrained_path, f_prefix, map_loc=self.device)                
                network.load_state_dict(state_dict, strict=True)
            except:
                print('Weights file: {:s} not available.'.format(dx_pretrained_path + '/' + f_prefix))
                exit(-3)
            
            self.networks_dx.append(network)

            # create network with correct device - dx (finetuned)
            network = ConvnextNetwork(weights=None, progress=False, n_classes=11, stochastic_depth_prob=0.1)
            network = network.to(device=self.device)
            
            # to evaluation mode
            network.eval()
            
            try: 
                model_path = folder + '/weights{:d}'.format(i) + '.h5'
                state_dict = torch.load(model_path, map_location=self.device)                
                network.load_state_dict(state_dict, strict=True)
            except:
                print('Weights file: {:s} not available.'.format(model_path))
                exit(-4)
            
            self.networks_dx_finetuned.append(network)

    # predict one image
    def predict_image(self, ecg_image, record):

        # resize image    
        ecg_image = resize_image(ecg_image)
                        
        # do not compute gradient
        with torch.no_grad():
            x = image_to_tensor(ecg_image, self.device)
            x = x.unsqueeze(0)
            #print('x.shape:', x.shape)
            if self.eval_mode:
                ecg_id = record
                if record.count('/') > 0:
                    ecg_id = ecg_id[ecg_id.rindex('/')+1:]
                if record.count('_') > 0:
                    ecg_id_as_int = int(ecg_id[:ecg_id.index('_')])
                else:
                    ecg_id_as_int = int(ecg_id)
                fold_df = self.ecg_folds.query("ECG_ID == @ecg_id_as_int")
                fold = 0
                if len(fold_df) > 0:
                    fold = fold_df.FOLD.iat[0]
                network = self.networks_rot[fold]
                avg_angle_n = network(x).cpu().numpy()[0,0]
            else:
                pred_list = [network(x).cpu().numpy()[0,0] for network in self.networks_rot]
                avg_angle_n = sum(pred_list) / len(pred_list)   
             
            # round to integer
            avg_angle = int(round(30 * avg_angle_n))
    
            # rotate image back
            ecg_image = rotate_image(ecg_image, -avg_angle)
    
            # to tensor
            x = image_to_tensor(ecg_image, self.device)
            x = x.unsqueeze(0)            
            #print('x.shape:', x.shape)
            
            if self.eval_mode:
                avg_prob1 = torch.sigmoid(self.networks_dx[fold](x)).cpu().numpy()[0,:]
                avg_prob2 = torch.sigmoid(self.networks_dx_finetuned[fold](x)).cpu().numpy()[0,:]
            else:
                prob_list1 = [torch.sigmoid(network(x)).cpu().numpy()[0,:] for network in self.networks_dx]
                avg_prob1 = sum(prob_list1) / len(prob_list1)
                
                prob_list2 = [torch.sigmoid(network(x)).cpu().numpy()[0,:] for network in self.networks_dx_finetuned]
                avg_prob2 = sum(prob_list2) / len(prob_list2)

        return self.w_pretrained * avg_prob1 + (1 - self.w_pretrained) * avg_prob2

    # predict all images for given record
    def predict(self, record):
        print('Record:', record) 
        
        # load ecg images
        ecg_images = load_images(record)
        
        # predict each image and average
        preds_for_imgs = [self.predict_image(ecg_image, record) for ecg_image in ecg_images]
        return sum(preds_for_imgs) / len(preds_for_imgs)