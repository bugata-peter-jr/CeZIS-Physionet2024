# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:59:48 2024

@author: bugatap
"""

import torch

import torch.nn.functional as F

class WeightedBCEWithLogitsLoss(torch.nn.modules.loss._Loss):
    def __init__(self, pos_weight=2.0, neg_weight=1.0, reduction: str='mean', label_smoothing=0.0):
        super(WeightedBCEWithLogitsLoss, self).__init__(reduction=reduction)
        self.label_smoothing = label_smoothing
        
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        
    def forward(self, output, target):
        #print('output:', output)
        #print('target:', target)
        weights = target * self.pos_weight + (1-target) * self.neg_weight
        #print('weights:', weights)
        # tym, ktore maju -1, dame vahu neg_weight
        bmask = torch.lt(target, torch.zeros_like(target))#.float()
        #print('bmask:', bmask)
        weights[bmask] *= 0
        #print('weights:', weights)
        if self.label_smoothing > 0.0:
            smooth_factor = torch.ones(target.shape, device=output.device) * self.label_smoothing
            smooth_factor[target==1] *= -1
            # smooth only 0 and 1 labels
            smooth_factor[torch.logical_and(target!=0, target!=1)] *= 0
            smooth_target = target + smooth_factor
        else:
            smooth_target = target
        result = F.binary_cross_entropy_with_logits(output, smooth_target, weight=weights, reduction=self.reduction)
        
        return result 
    
if __name__ == '__main__':
    
    # loss
    loss = WeightedBCEWithLogitsLoss()
    
    # loss test
    y = torch.Tensor([0,0,1,1,-1])
    p = torch.Tensor([0.05, 0.5, 0.05, 0.5, 0.5])
    
    result = loss(p, y)
    print('Loss:', result)