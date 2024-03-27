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

class FileImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, record_ids, y, mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225], random_erasing=False):
        self.path = path
        self.record_ids = record_ids
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
            
        images = load_images(self.path + '/' + record_id)
        
        if len(images) > 1:
            j = np.random.randint(0, len(images))
        else:
            j = 0
        image = images[j]
                    
        aug = iaa.Sequential([iaa.Resize({"width": 300, "height": "keep-aspect-ratio"}, interpolation='area')])
        image = aug(images=[np.array(image)])[0]
        image = Image.fromarray(image)
        image = image.convert('RGB')
        
        img_new = Image.new(image.mode, (300, 242), (255, 255, 255))
        img_new.paste(image)

        X = np.asarray(img_new, dtype=np.float32) 
        X = X[:, :, :3]        
                                
        # divide by 255
        X /= 255
        
        # image standardization
        X = (X - self.mean) / self.std
        
        X = torch.as_tensor(X, dtype=torch.float32)
        X = X.permute((2,0,1))

        if self.transform is not None:
            X = self.transform(X)
        
        y = self.y[i]
        y = torch.as_tensor(y, dtype=torch.float32)
        
        return X, y
