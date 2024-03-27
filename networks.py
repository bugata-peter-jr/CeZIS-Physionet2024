# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:39:56 2024

@author: bugatap
"""

import torch

from torchvision.models import convnext_tiny
    
class ConvnextNetwork(torch.nn.Module):
    def __init__(self, n_outputs=1):
        super().__init__()
        self.convnext = convnext_tiny()
        self.convnext.classifier[2] = torch.nn.Linear(768, n_outputs)
        
    def forward(self, x):
        return self.convnext(x)