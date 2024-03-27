# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 08:35:53 2024

@author: bugatap
"""

from inspect import getmembers

class Config(object):
    
    def __init__(self):

        # dataset and loader params
        self.P = 8
        
        # loss
        self.label_smoothing = 0.05        

        #                 NORM   CD  HYP   MI STTC
        self.pos_weight = [1.0, 1.5, 2.0, 1.5, 1.5]   
        self.neg_weight = [1.0, 1.0, 1.0, 1.0, 1.0]
        
        # mixup
        self.mixup = 0.0                                             # mixup

        # augmentations
        self.random_erasing = True
               
        # Train parameters  
        self.n_epochs = 5                                            # number of training epochs 
        self.batch_size = 64                                         # batch size
        self.max_lr = 0.0001                                         # maximal learning rate
        self.anneal_strategy = 'cos'                                 # anneal strategy
        self.pct_start = 0.3                                         # start pct
        self.div_final = 10000.0                                     # final div factor
        self.WD = 0.01 * 5                                           # weight decay
        self.use_16fp = False                                        # mixed precision
    
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