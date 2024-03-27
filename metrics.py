# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:00:54 2024

@author: bugatap
"""

import torch

# zakladna trieda - z nej dedia vsetky ostatne
class BaseMetric(object):
    
    def get_name(self):
        return 'base_metric'
    
    def compute(self, prediction, target):
        return 0
    

# multilabel F1 score - macro
class MultilabelF1Macro(BaseMetric):        
    def __init__(self, label_weights=None, n_classes=2, threshold=0.5, reverse_first_label=False):
        super(MultilabelF1Macro, self).__init__()
        # number of classes
        self.n_classes = n_classes
        # threshold for binary classification, i.e. if n_classes == 2
        self.threshold = threshold
        
        # label weights
        self.label_weights = label_weights
        
        # whether reverse the first label, 1 - label
        self.reverse_first_label = reverse_first_label 

    def get_name(self):
        return 'f1'

    def compute(self, prediction, target):
        if self.reverse_first_label:
            new_prediction = prediction.detach().clone()
            new_prediction[:,0] = 1 - new_prediction[:,0]
            new_target = target.detach().clone()
            new_target[:,0] = 1 - new_target[:,0]
            new_target[new_target == 2] = -1            
        else:
            new_prediction = prediction
            new_target = target
            
        #print('prediction:', prediction)
        #print('target:', target)
        
        classes = (new_prediction >= self.threshold)
            
        target_abs = torch.abs(new_target)

        # -1 and 0.5 will have weight 0
        bmask1 = torch.ge(new_target, torch.zeros_like(new_target))
        bmask2 = torch.ne(new_target, 0.5 * torch.ones_like(new_target))
        bmask = torch.logical_and(bmask1, bmask2).float()
        
        #print('bmask:', bmask)
            
        label_tps = ((classes == 1) & (target_abs == 1))
        label_tps = label_tps * bmask
        label_tp = label_tps.sum(axis=0)
        #print('TP:', label_tp)
        
        label_fps = ((classes == 1) & (target_abs == 0))
        label_fps = label_fps * bmask
        label_fp = label_fps.sum(axis=0)
        #print('FP:', label_fp)
        
        label_fns = ((classes == 0) & (target_abs == 1))
        label_fns = label_fns * bmask
        label_fn = label_fns.sum(axis=0)
        #print('FN:', label_fn)
        
        result = (label_tp)/(label_tp + (label_fp + label_fn)/2)

        if self.label_weights is None:
            f1 = result.sum() / classes.size(1)
        else:
            f1 = (result * self.label_weights).sum() / self.label_weights.sum()
            
        return f1.item() * 100
