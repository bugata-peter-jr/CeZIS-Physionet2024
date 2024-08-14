# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 08:35:53 2024

@author: bugatap
"""

from inspect import getmembers


from torchvision.models import ConvNeXt_Tiny_Weights

class Config(object):
    
    def __init__(self):
        # ConvNext config
        self.progress = False
        #self.pretrained = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.pretrained = None
        self.stochastic_depth_prob = 0.1

        # dataset and loader params
        self.data_path = '/projects/ECG/python/Physionet2024/official_phase/demo/train_images_lr'
        self.P = 8
        
        # loss
        self.label_smoothing = 0.0
        
        self.classes = ['NORM', 'Acute MI', 'Old MI', 'STTC', 'CD', 'HYP', 'PAC', 'PVC', 'AFIB/AFL', 'TACHY', 'BRADY']
    
        self.pos_weight = [1.0, 4.0, 2.0, 1.5, 2.0, 2.0, 3.0, 2.0, 1.0, 1.0, 1.0] 
        self.neg_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        
        # mixup
        self.mixup = 0.0                                             # mixup

        # augmentations
        self.random_erasing = True
               
        # Train parameters  
        self.n_epochs = 5                                            # number of training epochs 
        self.batch_size = 64                                         # batch size
        self.accum_iters = 1                                         # possible gradient accumulation
        self.optimizer_type = 'adamw'                                # typ optimalizatora
        self.scheduler_type = 'one_cycle'                            # typ planovaca lr        
        self.max_lr = 0.0001                                         # maximal learning rate
        self.anneal_strategy = 'cos'                                 # anneal strategy
        self.pct_start = 0.3                                         # start pct
        self.div_final = 10000.0                                     # final div factor
        self.WD = 0.01 * 5                                           # weight decay
        self.use_16fp = True                                         # mixed precision
                
        self.use_4v_logic = True
    
        # weight of pretrained model
        self.w_pretrained = 0.0    
    
    def __str__(self):
        s = ''
        for name, value in getmembers(self):
            if name.startswith('__'):
                continue
            s += name + ' : ' + str(value) + '\n'
        return s
    
if __name__ == '__main__':
    
    cfg = Config()
    print(cfg)