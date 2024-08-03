# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:40:48 2024

@author: bugatap
"""


import torch

from torchvision.models import convnext_tiny
from torch.nn import functional as F

class LayerNorm2d(torch.nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class CombinedPool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.maxpool = torch.nn.AdaptiveMaxPool2d(1)
       
    def forward(self, x):
        return self.avgpool(x), self.maxpool(x) 
       

class Classifier(torch.nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.ln_avgp = LayerNorm2d((768,))
        self.ln_mp = LayerNorm2d((768,))
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(2*768, n_classes)
        
    def forward(self, x):
        x_avgp, x_mp = x
        x_avgp = self.ln_avgp(x_avgp)
        x_mp = self.ln_mp(x_mp)
        #print(x_avgp.shape, x_mp.shape)
        x = torch.cat([x_avgp, x_mp], dim=1)
        #print(x.shape)
        x = self.flatten(x)
        x = self.linear(x)
        return x
    
class ConvnextNetwork(torch.nn.Module):
    def __init__(self, weights, progress, n_classes, stochastic_depth_prob=0.1):
        super().__init__()
        if weights is not None:
            self.convnext = convnext_tiny(weights=weights, progress=progress, stochastic_depth_prob=stochastic_depth_prob)
        else:
            self.convnext = convnext_tiny(stochastic_depth_prob=stochastic_depth_prob)            
        self.convnext.avgpool = CombinedPool()
        self.convnext.classifier = Classifier(n_classes)
        
    def forward(self, x):
        return self.convnext(x)
    
class ConvnextNetworkSimple(torch.nn.Module):
    def __init__(self, weights, progress, n_classes, stochastic_depth_prob=0.1):
        super().__init__()
        if weights is not None:
            self.convnext = convnext_tiny(weights=weights, progress=progress, stochastic_depth_prob=stochastic_depth_prob)
        else:
            self.convnext = convnext_tiny(stochastic_depth_prob=stochastic_depth_prob)                        
        self.convnext.classifier[2] = torch.nn.Linear(768, n_classes)
        
    def forward(self, x):
        return self.convnext(x)

    
