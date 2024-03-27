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

from helper_code import *

import pandas as pd

import torch
from networks import ConvnextNetwork
from train_convnext import train

from imgaug import augmenters as iaa
from PIL import Image

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your digitization model.
def train_digitization_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training of digitization model not implemented.')

    return

# Train your dx classification model.
def train_dx_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training of dx model...')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    pretrained_path = './pretrained/dx'
    train(data_folder, pretrained_path, model_folder)

    return

# Load your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you do not train a digitization model, then you can return None.
def load_digitization_model(model_folder, verbose):
    return None

# Load your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function. If you do not train a dx classification model, then you can return None.
def load_dx_model(model_folder, verbose):
    model = Model(eval_mode=False)
    model.load(model_folder)
    return model

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function.
def run_digitization_model(digitization_model, record, verbose):

    # Load the dimensions of the signal.
    header_file = get_header_file(record)
    header = load_text(header_file)

    num_samples = get_num_samples(header)
    num_signals = get_num_signals(header)

    return np.zeros(shape=(num_samples, num_signals), dtype=np.int16)

# Run your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function.
def run_dx_model(dx_model, record, signal, verbose):

    probabilities = dx_model.predict(record)
    classes = dx_model.classes

    # Choose the class(es) with the highest probability as the label(s).
    max_probability = np.nanmax(probabilities)
    labels = [classes[i] for i, probability in enumerate(probabilities) if probability == max_probability]

    return labels

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################
    
class Model(object):
    
    def __init__(self, eval_mode=False):
        
        self.networks_rot = []
        
        self.networks_dx = []
        
        self.eval_mode = eval_mode
        if self.eval_mode:
            self.ecg_folds = pd.read_csv('./folds.csv')
        
        # device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.classes = ['NORM','CD','HYP','MI','STTC']
        
        
    def load(self, folder):
        pretrained_folder = './pretrained'
                
        # create and load networks 
        for i in range(5):
            fname = 'weights{:d}.h5'.format(i)
            
            # create network with correct device - rotation
            network = ConvnextNetwork(n_outputs=1)
            network = network.to(device=self.device)
            
            # to evaluation mode
            network.eval()
            
            try:
                model_file = pretrained_folder + '/rotation/' + fname
                network.load_state_dict(torch.load(model_file), strict=True)
            except:
                print('Weights file: {:s} not available.'.format(model_file))
            
            self.networks_rot.append(network)

            # create network with correct device - dx
            network = ConvnextNetwork(n_outputs=5)
            network = network.to(device=self.device)
            
            # to evaluation mode
            network.eval()
            
            try:
                model_file = pretrained_folder + '/dx/' + fname                
                network.load_state_dict(torch.load(model_file), strict=True)
            except:
                print('Weights file: {:s} not available.'.format(model_file))
            
            self.networks_dx.append(network)
            
    def predict_image(self, ecg_image, record):
        # load image              
        
        aug = iaa.Sequential([iaa.Resize({"width": 300, "height": "keep-aspect-ratio"}, interpolation='area')])
        ecg_image = aug(images=[np.array(ecg_image)])[0]
        ecg_image = Image.fromarray(ecg_image)
        ecg_image = ecg_image.convert('RGB')
        
        #img_new = Image.new(ecg_image.mode, (300, 242), (255, 255, 255))
        #img_new.paste(ecg_image)

        ecg_image = np.asarray(ecg_image, dtype=np.float32) 
        ecg_image = ecg_image[:, :, :3]        
                                
        # divide by 255
        ecg_image /= 255
        
        # normalization
        mean = np.array([0.485,0.456,0.406])
        std = np.array([0.229, 0.224, 0.225])
        ecg_image = (ecg_image - mean) / std                
        
        # do not compute gradient
        with torch.no_grad():
            # to tensor
            x = torch.as_tensor(ecg_image, dtype=torch.float32, device=self.device)
            x = x.permute((2,0,1))
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
            aug = iaa.Sequential([iaa.Affine(rotate=-avg_angle)])
            ecg_image = aug(images=[np.array(ecg_image)])[0]
    
            # to tensor
            x = torch.as_tensor(ecg_image, dtype=torch.float32, device=self.device)
            x = x.permute((2,0,1))
            x = x.unsqueeze(0)
            #print('x.shape:', x.shape)
            if self.eval_mode:
                avg_prob = torch.sigmoid(self.networks_dx[fold](x)).cpu().numpy()[0,:]
            else:
                prob_list = [torch.sigmoid(network(x)).cpu().numpy()[0,:] for network in self.networks_dx]
                avg_prob = sum(prob_list) / len(prob_list)

        return avg_prob
    
    def predict(self, record):
        print('Record:', record) 
        
        # load ecg images
        ecg_images = load_image(record)
        
        # predict each image and average
        preds_for_imgs = [self.predict_image(ecg_image, record) for ecg_image in ecg_images]
        return sum(preds_for_imgs) / len(preds_for_imgs)