# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:32:14 2024

@author: bugatap
"""

import numpy as np

from helper_code import load_images
from PIL import Image

import torch
from torchvision.transforms import RandomErasing
from imgaug import augmenters as iaa

def resize_image(ecg_image):
    ecg_image = ecg_image.convert('RGB')
    
    aug = iaa.Sequential([iaa.Resize({"width": 600, "height": "keep-aspect-ratio"}, interpolation='area')])
    ecg_image = aug(images=[np.array(ecg_image)])[0]        
    ecg_image = Image.fromarray(ecg_image)
    return ecg_image            

def rotate_image(ecg_image, angle):
    aug = iaa.Sequential([iaa.Affine(rotate=angle)])
    ecg_image = aug(images=[np.array(ecg_image)])[0]
    ecg_image = Image.fromarray(ecg_image)
    return ecg_image            

def image_to_tensor(ecg_image, device=None):
    X = np.array(ecg_image)
           
    # divide by 255
    X = X / 255
    
    # normalization
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229, 0.224, 0.225])
    X = (X - mean) / std
    
    if device is None:
        X = torch.as_tensor(X, dtype=torch.float32)
    else:
        X = torch.as_tensor(X, dtype=torch.float32, device=device)
        
    X = X.permute((2,0,1))
    return X


class FileImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, record_ids, rotations, y, mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225], random_erasing=False):
        self.path = path
        self.record_ids = record_ids
        self.rotations = rotations

        self.y = y
        
        self.mean = np.array(mean)
        self.std = np.array(std)
        
        self.transform = None
        if random_erasing:
            self.transform = RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
        
    def __len__(self):
        return len(self.record_ids)

    def __getitem__(self, i):
        
        record_id = self.record_ids[i]
        #print(record_id)
         
        images = load_images(self.path + '/' + record_id)
        
        if len(images) > 1:
            j = np.random.randint(0, len(images))
        else:
            j = 0
        image = images[j]
        
        rotate = -self.rotations.get(str(record_id) + '-' + str(j))

        image = resize_image(image)     
           
        img_new = Image.new(image.mode, (600, 512), (255, 255, 255))
        img_new.paste(image)

        img_new = rotate_image(img_new, rotate)

        X = image_to_tensor(img_new)                                        
        
        if self.transform is not None:
            X = self.transform(X)
        
        y = self.y[i]
        y = torch.as_tensor(y, dtype=torch.float32)
        
        return X, y
    
