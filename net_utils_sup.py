# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 18:46:11 2022

@author: bugatap
"""


import time

import torch
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR
import torch.distributed as dist

import numpy as np

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        #print('set_eval:', classname)
        m.eval()        

class NetworkModel(object):
    # constructor
    def __init__(self, network, optimizer, scheduler, 
                 loss_function, metrics, use_metric=False, accum_iters=1,
                 n_repeats=1, pred_agg='average',
                 verbose=False, grad_norm=None, use_16fp=False, 
                 freeze_BN=False, output_fn=None, mixup=0.0, 
                 rank=None, world_size=2):
        # trained network - object of type  torch.nn.Module
        self.network = network
        # optimizer - object from torch.nn.optim package
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # loss function - object of LossFunction class
        self.loss_function = loss_function
        
        # metric - e.g. accuracy
        self.metrics = metrics
        if self.metrics is not None and not isinstance(self.metrics, list):
            self.metrics = [self.metrics]
        if self.metrics == []:
            self.metrics = None
        
        # criterion to use for model selection
        # metric or loss
        self.use_metric = use_metric
        
        # accum step
        self.accum_iters = accum_iters
        
        # verbose mode
        self.verbose = verbose

        # gradient clipping to L2 norm grad_norm
        self.grad_norm = grad_norm
        
        # number of repeats used for model validation
        self.n_repeats = n_repeats

        # aggregation of preds
        self.pred_agg = pred_agg
        
        # boolean flag whether to use 16bit precision
        self.use_16fp = use_16fp
        
        # boolean flag whether to freeze BN layers
        self.freeze_BN = freeze_BN 
        
        # output function
        self.output_fn = output_fn
        
        # optional mixup
        self.mixup = mixup
        
        # rank
        self.rank = rank
        self.world_size = world_size
        
    # fit - model training
    def fit(self, loader, loader_valid, n_epochs, model_file):        
        # optimizer and loss
        optimizer = self.optimizer
        cost = self.loss_function
        
        # get the current device
        device = next(self.network.parameters()).device
        print('Rank: {} device: {}'.format(self.rank, device))

        # get the current datatype
        dtype = next(self.network.parameters()).dtype
        
        # best validation loss
        best_loss_valid = None
        
        # best metric on validation set
        best_measure_valid = None
        
        #torch.autograd.set_detect_anomaly(True)
        
        if self.use_16fp:
            grad_scaler = torch.cuda.amp.GradScaler()
                    
        # training        
        for t in range(n_epochs):
            # when using distributed data parallel
            if isinstance(self.network, torch.nn.parallel.DistributedDataParallel):
                loader.sampler.set_epoch(t)
            
            preds_raw = None
            tgt = None
            
            t1 = time.time()

            #print('Epoch:', t)

            # switch the network to training mode
            self.network.train()                        
            
            if self.freeze_BN:
                #print('Freeze BN')
                self.network.apply(set_bn_eval)
            
            best_model_flag = ' '
            i = 0
            #print('Before Loader loop', flush=True)
            for data, target in loader:
        
                # prediction
                data = data.to(device=device, dtype=dtype)
                if isinstance(self.loss_function, torch.nn.CrossEntropyLoss):
                    target = target.to(device=device, dtype=torch.long)
                else:
                    target = target.to(device=device, dtype=dtype)
                    
                #print('Rank: {} data.device: {} target.device: {} data.shape: {}'.format(self.rank, data.device, target.device, data.shape), flush=True)
                    
                # mixup (optional)
                if self.mixup > 0.0:
                    with torch.no_grad():
                        lambd = np.random.beta(self.mixup, self.mixup)
                        permutation = torch.randperm(data.size(0))
                        data = lambd * data + (1 - lambd) * data[permutation]
                        target = lambd * target + (1 - lambd) * target[permutation]

                # automatic mixed precision support
                if self.use_16fp:
                    with torch.cuda.amp.autocast():
                        prediction_raw = self.network(data)
                                            
                        # loss computation
                        loss = cost(prediction_raw, target)
                else:
                    prediction_raw = self.network(data)
                    #print('Rank: {} After prediction'.format(self.rank), flush=True)    
                    #time.sleep(20)                    
                                        
                    # loss computation
                    loss = cost(prediction_raw, target)
                    #print('Rank: {} After cost'.format(self.rank), flush=True)    
                    #time.sleep(20)                    
                    
                # when to apply optimizer step
                if self.accum_iters > 1:
                    #loss /= self.accum_iters
                    apply_optimizer_step = (i + 1) % self.accum_iters == 0 or (i + 1) == len(loader)
                else:
                    apply_optimizer_step = True    

                # cleaning gradient
                if apply_optimizer_step:
                    #print('Optimizer step:', i + 1)
                    optimizer.zero_grad(set_to_none=True)

                # automatic mixed precision support
                if self.use_16fp:
                    grad_scaler.scale(loss).backward()

                    # gradient clipping by L2 norm
                    if self.grad_norm is not None:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_norm, 2)
                        
                    if apply_optimizer_step:
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    
                else:
                    loss.backward()
                    #print('Rank: {} After backward'.format(self.rank), flush=True)    
                    #time.sleep(20)                    
                    
                    # gradient clipping by L2 norm
                    if self.grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_norm, 2)
                    
                    # optimization step
                    if apply_optimizer_step:
                        optimizer.step()
                        #print('Rank: {} After optimizer'.format(self.rank), flush=True)    
                        #time.sleep(20)                    

                # apply batch learning rate scheduler
                if self.scheduler is not None and isinstance(self.scheduler, (CyclicLR, OneCycleLR)):
                    self.scheduler.step()
                    #print('Rank: {} After scheduler'.format(self.rank), flush=True)    
                    #time.sleep(20)                    

                # saving prediction and target for evaluation
                with torch.no_grad():
                    if preds_raw is None:
                        preds_raw = prediction_raw
                        tgt = target
                    else:
                        preds_raw = torch.cat((preds_raw, prediction_raw))
                        tgt = torch.cat((tgt, target))
                    i += 1

            #gc.collect() 
            #time.sleep(1)
            with torch.no_grad():
                # measure on train set
                preds = preds_raw if self.output_fn is None else self.output_fn(preds_raw)
                if self.metrics is not None:
                    measures_avg = [metric.compute(preds, tgt) for metric in self.metrics]
    
                # loss on train set
                loss_avg = cost(preds_raw, tgt) 
                loss_avg = loss_avg.item()                    
            
                if isinstance(self.network, torch.nn.parallel.DistributedDataParallel):
                    per_process_losses = [None for _ in range(self.world_size)]        
                    #print('Rank: {}, before all_gather_object: {}'.format(self.rank, loss_avg), flush=True)
                    dist.all_gather_object(per_process_losses, loss_avg) 
                    loss_avg = sum(per_process_losses) / len(per_process_losses)
                    #print('Rank: {}, after all_gather_object: {}'.format(self.rank, loss_avg), flush=True)
                                                
            # validation after epoch
            if loader_valid is not None: # and t+1 > n_epochs/2:
                if self.metrics is None:
                    loss_valid_avg = self.eval(loader_valid, silent=True)
                else:
                    loss_valid_avg, measures_valid_avg = self.eval(loader_valid, silent=True)

                with torch.no_grad():                 
                    # saving the best model according to loss or metric
                    if model_file is not None:
                        if self.use_metric is not None:
                            metric_names = [eval_metric.get_name() for eval_metric in self.metrics]
                            idx = metric_names.index(self.use_metric)
                            measure_valid_avg = measures_valid_avg[idx]
                            if best_measure_valid is None or measure_valid_avg > best_measure_valid:
                                best_measure_valid = measure_valid_avg
                                torch.save(self.network.state_dict(), f=model_file)
                                best_model_flag = '*'                
                        else:    
                            if best_loss_valid is None or loss_valid_avg < best_loss_valid:
                                best_loss_valid = loss_valid_avg
                                torch.save(self.network.state_dict(), f=model_file)
                                best_model_flag = '*'                            
                    
            t2 = time.time()                

            # apply epoch learning rate scheduler
            if self.scheduler is not None and not isinstance(self.scheduler, (CyclicLR, OneCycleLR)):
                self.scheduler.step()

            if self.verbose:
                if self.metrics is None:
                    if loader_valid is not None:
                        print('Epoch: {0:04d}/{1:04d}{4:1s} ({5:4.2f} s) loss train avg: {2:6.4f}   loss valid: {3:6.4f}'
                              .format(t, n_epochs, loss_avg, loss_valid_avg, best_model_flag, t2-t1), flush=True)
                    else:
                        print('Epoch: {0:04d}/{1:04d} ({3:4.2f} s) loss train avg: {2:6.4f}'
                              .format(t, n_epochs, loss_avg, t2-t1), flush=True)
                else:
                    metric_names = [eval_metric.get_name() for eval_metric in self.metrics]
                    metric_names_display = '/'.join(metric_names)
                    metric_values_display = '/'.join(['{:4.2f}'.format(v) for v in measures_avg])
                    if loader_valid is not None:
                        metric_values_v_display = '/'.join(['{:4.2f}'.format(v) for v in measures_valid_avg])
                        print('Epoch: {0:04d}/{1:04d}{6:1s} ({8:4.2f} s) loss train avg: {2:8.6f}   {7:s} train avg: {3:s}   loss valid: {4:8.6f}   {7:s} valid: {5:s}'
                              .format(t, n_epochs, loss_avg, metric_values_display, loss_valid_avg, metric_values_v_display, best_model_flag, metric_names_display, t2-t1), flush=True)
                    else:
                        print('Epoch: {0:04d}/{1:04d} ({5:4.2f} s) loss train avg: {2:6.4f}   {4:s} train avg: {3:s}'
                              .format(t, n_epochs, loss_avg, metric_values_display, metric_names_display, t2-t1), flush=True)  


            
        return None
    
    # prediction
    def predict_unagg(self, loader, return_target=False, return_raw=False):
        # get the current device
        device = next(self.network.parameters()).device

        # get the current datatype
        dtype = next(self.network.parameters()).dtype

        # switch the network to evaluation mode
        network = self.network
        network.eval()

        with torch.no_grad():
            preds = None
            preds_raw = None
            tgt = None
            
            for data, target in loader:
        
                # prediction
                data = data.to(device=device, dtype=dtype)
                if isinstance(self.loss_function, torch.nn.CrossEntropyLoss):
                    target = target.to(device=device, dtype=torch.long)
                else:
                    target = target.to(device=device, dtype=dtype)
                    
                pred_raw = network(data)
                
                if self.output_fn is not None:
                    pred = self.output_fn(pred_raw)
                else:
                    pred = pred_raw
                
                if preds is None:
                    preds = pred
                    preds_raw = pred_raw
                    tgt = target
                else:
                    preds = torch.cat((preds, pred))
                    preds_raw = torch.cat((preds_raw, pred_raw))
                    tgt = torch.cat((tgt, target))
                
        # result
        result = [preds]
        if return_raw:
            result.append(preds_raw)
        if return_target:
            result.append(tgt)
        
        return result
    
    # helper function for aggregate predictions
    def aggregate_preds(self, preds, tgt):
        # more repeats for validation
        if self.n_repeats > 1:
            n_reps = self.n_repeats
            pred_list = [preds[torch.arange(n_reps,preds.size(0)+n_reps,n_reps)-r-1] for r in range(n_reps)]
            y_pred = pred_list[0]
            y = tgt[torch.arange(n_reps,preds.size(0)+n_reps,n_reps)-1]
            for r in range(1, n_reps):
                if self.pred_agg == 'max':
                    y_pred = torch.maximum(y_pred, pred_list[r])
                elif self.pred_agg == 'average':
                    y_pred += pred_list[r]
            if self.pred_agg == 'average':
                y_pred /= self.n_repeats
        else:
            y_pred = preds
            y = tgt
        return y_pred, y

    def predict(self, loader, to_numpy=True, only_prediction=True):
        y_pred, y_pred_raw, tgt = self.predict_unagg(loader, return_target=True, return_raw=True)
        y_pred, tgt = self.aggregate_preds(y_pred, tgt)
        result = [y_pred, y_pred_raw, tgt]
        
        if to_numpy:
            result = [x.cpu().data.numpy() for x in result]

        if only_prediction:
            result = result[0]
        
        return result
    
    # evaluation - possibly in more epochs
    # basic version is only once
    def eval(self, loader, n_epochs=1, silent=True):
        # switch the network to evaluation mode
        self.network.eval()
        
        cost = self.loss_function
                
        test_losses = np.zeros(n_epochs)
        if self.metrics is not None:
            test_measures = np.zeros(shape=(n_epochs, len(self.metrics)))
        
        with torch.no_grad():   
            for t in range(1, n_epochs+1):
                # get prediction and target                               
                y_pred, y_pred_raw, tgt = self.predict_unagg(loader, return_target=True, return_raw=True)
                                                        
                j = t - 1
                loss = cost(y_pred_raw, tgt).item()
                
                if isinstance(self.network, torch.nn.parallel.DistributedDataParallel):
                    per_process_losses = [None for _ in range(self.world_size)]        
                    #print('Rank: {}, before all_gather_object: {}'.format(self.rank, loss_avg), flush=True)
                    dist.all_gather_object(per_process_losses, loss) 
                    loss = sum(per_process_losses) / len(per_process_losses)
                    #print('Rank: {}, after all_gather_object: {}'.format(self.rank, loss_avg), flush=True)                
                
                test_losses[j] = loss
                if self.metrics is not None:
                    y_pred, tgt = self.aggregate_preds(y_pred, tgt)
                    #print([metric.compute(y_pred, tgt) for metric in self.metrics])
                    test_measures[j] = [metric.compute(y_pred, tgt) for metric in self.metrics]
                if self.verbose and not silent:
                    if self.metrics is not None:
                        metric_names = [eval_metric.get_name() for eval_metric in self.metrics]
                        metric_names_display = '/'.join(metric_names)
                        metric_values_display = '/'.join(['{:4.2f}'.format(v) for v in test_measures[j]])
                        print('Epoch: {0:04d}/{1:04d} loss avg: {2:6.4f}   {4:s} avg: {3:s}'
                              .format(t, n_epochs, test_losses[j], metric_values_display, metric_names_display), flush=True)
                    else:
                        print('Epoch: {0:04d}/{1:04d} loss avg: {2:6.4f}'
                              .format(t, n_epochs, test_losses[j]), flush=True)                        
      
                    
        # if is model evaluated once, return only one element
        if n_epochs == 1:
            if self.metrics is None:
                return test_losses[0]
            return test_losses[0], test_measures[0,:].tolist()
        
        if self.metrics is None:
            return test_losses
        return test_losses, test_measures
    
    # load weights
    def load_weights(self, model_file, strict=False):
        #self.network = torch.load(model_file)
        state_dict = torch.load(model_file)
        self.network.load_state_dict(state_dict, strict)

    def save_weights(self, model_file):
        torch.save(self.network.state_dict(), f=model_file)
    
