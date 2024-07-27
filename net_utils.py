# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 12:58:28 2024

@author: bugatap
"""

import time

import torch
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR
import torch.distributed as dist

import numpy as np

from pytorch_model_summary import summary

from typing import Dict, Any, Union

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        #print('set_eval:', classname)
        m.eval()        

def singleton_to_list(obj):
    if isinstance(obj, (list, tuple)):
        return obj
    return [obj]

def unpack_singleton(obj_list):
    if len(obj_list) == 1:
        return obj_list[0]
    return obj_list

def remove_prefix_from_state_dict_if_exists(state_dict: Union[Dict[str, Any], Any],
                                            prefix: str) -> Union[Dict[str, Any], Any]:
    """This function strips a prefix if it exists in the model
    state_dict and metadata.
    Args:
        state_dict : DP/DDP pytorch model state_dict
        prefix (str) : Prefix to be removed from the DP/DDP model state_dict to
                       to make it compatible with regular pytorch model.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return state_dict

    for key in list(state_dict.keys()):
        new_key = key[len(prefix):]
        state_dict[new_key] = state_dict.pop(key)

    try:
        metadata = state_dict._metadata
    except AttributeError:
        pass

    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)
    return state_dict

class NetworkModel(object):
    # constructor
    def __init__(self, network, optimizer, scheduler, 
                 loss_fn, loss_weights=None, metric_fn=None, use_metric=False, accum_iters=1,
                 n_repeats=1, pred_agg=None,
                 verbose=False, grad_norm=None, use_16fp=False, 
                 freeze_BN=False, output_fn=None, mixup=0.0, 
                 rank=None, world_size=2):
        # trained network - object of type  torch.nn.Module
        self.network = network
        # optimizer - object from torch.nn.optim package
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # loss function - object of LossFunction class
        self.loss_functions = singleton_to_list(loss_fn)
        
        if loss_weights is None:
            loss_weights = np.ones(len(self.loss_functions))
        self.loss_weights = loss_weights
        
        # metric: list of lists - e.g. accuracy
        self.metrics = []
        if metric_fn is not None:
            for metric in metric_fn:
                metric = singleton_to_list(metric)
                self.metrics.append(metric)
        
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
        self.output_fn = singleton_to_list(output_fn)
        
        # optional mixup
        self.mixup = mixup
        
        # rank
        self.rank = rank
        self.world_size = world_size
        
        self.val_preds = None

    # unpack and prepare batch
    def prepare_batch(self, batch):
        #for b in batch:
        #    print(type(b))
        
        x, y, w = None, None, None
        if not isinstance(x, (list, tuple)):
            x = batch
        if len(batch) == 1:
            x, = batch
        if len(batch) == 2:
            x, y = batch
        if len(batch) == 3:
            x, y, w = batch

        # get the current device
        device = next(self.network.parameters()).device

        # get the current datatype
        dtype = next(self.network.parameters()).dtype
        
        x = singleton_to_list(x)
        x = [x_.to(device, dtype) for x_ in x]
        if y is not None:
            y = singleton_to_list(y)
            y = [y_.to(device, dtype) for y_ in y]
        if w is not None:
            w = singleton_to_list(w)
            w = [w_.to(device, dtype) for w_ in w]
            
        return x, y, w

    # fit - model training
    def fit(self, loader, loader_valid, n_epochs, model_file):
        # best error/score on val. set
        best_val_score = None
        best_val_error = None

        if self.use_16fp:
            grad_scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(n_epochs):
            # when using distributed data parallel
            if isinstance(self.network, torch.nn.parallel.DistributedDataParallel):
                loader.sampler.set_epoch(epoch)
            
            preds_raw = [None] * len(self.loss_functions)
            tgt = [None] * len(self.loss_functions)
            
            best_model_flag = ''
            start = time.time()
                
            # to train mode
            self.network.train()
            
            # freeze bn
            if self.freeze_BN:
                self.network.apply(set_bn_eval)
            
            # iterate train loader
            for step, batch in enumerate(loader):
                # batch is x and y
                x, y, _ = self.prepare_batch(batch)
                
                #x, y = batch
                #x1, x2 = x
                #print('Data shape:', x1.shape, x2.shape, 'target.shape:', y.shape)
                
                # mixup
                if self.mixup > 0.0:
                    with torch.no_grad():
                        lambd = np.random.beta(self.mixup, self.mixup)
                        permutation = torch.randperm(x.size(0))
                        x = [lambd * x_ + (1 - lambd) * x_[permutation] for x_ in x]
                        y = [lambd * y_ + (1 - lambd) * y_[permutation] for y_ in y]                    
                    
                # computing loss
                # automatic mixed precision support
                if self.use_16fp:
                    with torch.cuda.amp.autocast(): 
                        outputs = singleton_to_list(self.network(*x))
                                            
                        # loss computation
                        losses = [self.loss_weights[task_id] * self.loss_functions[task_id](outputs[task_id], y[task_id]) for task_id in range(len(self.loss_functions))] 
                        total_loss = sum(losses)
                else:
                    outputs = singleton_to_list(self.network(*x))                   
                                        
                    # loss computation
                    losses = [self.loss_weights[task_id] * self.loss_functions[task_id](outputs[task_id], y[task_id]) for task_id in range(len(self.loss_functions))] 
                    total_loss = sum(losses)
                                       
                # when to apply optimizer step
                if self.accum_iters > 1:
                    #loss /= self.accum_iters
                    apply_optimizer_step = (step + 1) % self.accum_iters == 0 or (step + 1) == len(loader)
                else:
                    apply_optimizer_step = True    

                # cleaning gradient
                if apply_optimizer_step:
                    self.optimizer.zero_grad(set_to_none=True)

                # automatic mixed precision support
                if self.use_16fp:
                    grad_scaler.scale(total_loss).backward()

                    # gradient clipping by L2 norm
                    if self.grad_norm is not None:
                        grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_norm, 2)
                        
                    if apply_optimizer_step:
                        grad_scaler.step(self.optimizer)
                        grad_scaler.update()
                    
                else:
                    total_loss.backward()
                    #print('Rank: {} After backward'.format(self.rank), flush=True)    
                    #time.sleep(20)                    
                    
                    # gradient clipping by L2 norm
                    if self.grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_norm, 2)
                    
                    # optimization step
                    if apply_optimizer_step:
                        self.optimizer.step()
                        #print('Rank: {} After optimizer'.format(self.rank), flush=True)    
                        #time.sleep(20)                  
                
                # batch scheduler
                if self.scheduler is not None and isinstance(self.scheduler, (CyclicLR, OneCycleLR)):
                    if apply_optimizer_step:
                        self.scheduler.step()
               
                # gathering preds and targets
                # network has possibly more outputs
                with torch.no_grad():
                    for task_id in range(len(self.loss_functions)):
                        if preds_raw[task_id] is None:
                            preds_raw[task_id] = outputs[task_id].to(device='cpu')
                            tgt[task_id] = y[task_id].to(device='cpu')
                        else:
                            preds_raw[task_id] = torch.cat((preds_raw[task_id], outputs[task_id].to(device='cpu')))
                            tgt[task_id] = torch.cat((tgt[task_id], y[task_id].to(device='cpu'))) 
                            
                #break                
                
                if self.verbose and step != 0 and step % 1000 == 0:
                    end_local = time.time()
                    duration = (end_local - start) / step
                    losses_avg = [self.loss_functions[task_id](preds_raw[task_id], tgt[task_id]).item() for task_id in range(len(self.loss_functions))]
                    losses_avg_str = '/'.join(['{:6.4f}'.format(loss_avg) for loss_avg in losses_avg])
                    #print('Step:', step, 'time:', duration, 'loss:', losses_avg_str, flush=True)
                    print('batch: {0:06d}/{1:06d} ({2:4.2f} s) loss train avg: {3:s}'.format(step, len(loader), duration, losses_avg_str), flush=True)


            # compute error and metrics on train set
            with torch.no_grad():
                losses_avg = [self.loss_functions[task_id](preds_raw[task_id], tgt[task_id]).item() for task_id in range(len(self.loss_functions))]
                if isinstance(self.network, torch.nn.parallel.DistributedDataParallel):
                    for task_id in range(len(self.loss_functions)):
                        per_process_losses = [None for _ in range(self.world_size)]        
                        #print('Rank: {}, before all_gather_object: {}'.format(self.rank, loss_avg), flush=True)
                        dist.all_gather_object(per_process_losses, losses_avg[task_id]) 
                        losses_avg[task_id] = sum(per_process_losses) / len(per_process_losses)
                
                
                if self.metrics is not None:
                    # apply output function
                    predictions = list(preds_raw)
                    for task_id in range(len(self.loss_functions)):
                        if self.output_fn[task_id] is not None:
                            predictions[task_id] = self.output_fn[task_id](predictions[task_id])
    
                    measures_avg = [metric.compute(predictions[task_id], tgt[task_id]) for task_id in range(len(self.loss_functions)) for metric in self.metrics[task_id]]


            # epoch scheduler
            if self.scheduler is not None and not isinstance(self.scheduler, (CyclicLR, OneCycleLR)):
                self.scheduler.step()
                
            # validation or model selection
            if loader_valid is not None:
                self.val_preds, losses_valid_avg, measures_valid_avg = self.predict_and_eval(loader_valid)
                loss_valid_avg = sum(losses_valid_avg)
                    
                if self.use_metric is None or measures_valid_avg is None:
                    if best_val_error is not None or loss_valid_avg < best_val_error:
                        best_val_error = loss_valid_avg
                        best_model_flag = '*'
                        if model_file is not None:
                            self.save_weights(model_file)
                else:
                    if len(self.loss_functions) == 1:
                        metric_names = [metric.get_name() for task_id in range(len(self.loss_functions)) for metric in self.metrics[task_id]]
                    else:
                        metric_names = ['o{:d}_'.format(task_id) + metric.get_name() for task_id in range(len(self.loss_functions)) for metric in self.metrics[task_id]]
                    idx = metric_names.index(self.use_metric)
                    score = measures_valid_avg[idx]
                    if best_val_score is None or score > best_val_score:
                        best_val_score = score
                        best_model_flag = '*'
                        if model_file is not None:
                            self.save_weights(model_file)
                            
            end = time.time()
            
            #dev = torch.cuda.current_device()
            #print('Device:', dev, 'Allocated memory:', torch.cuda.memory_allocated(dev) // 1024, flush=True)
                            
            # verbose
            if self.verbose:
                losses_avg_str = '/'.join(['{:6.4f}'.format(loss_avg) for loss_avg in losses_avg])
                if self.metrics is None:
                    if loader_valid is not None:
                        losses_avg_str_v = '/'.join(['{:6.4f}'.format(loss_avg) for loss_avg in losses_valid_avg])
                        output_str = 'Epoch: {0:04d}/{1:04d}{4:1s} ({5:4.2f} s) loss train avg: {2:s}   loss valid: {3:s}'.format(epoch+1, n_epochs, losses_avg_str, loss_valid_avg, best_model_flag, end-start)
                    else:
                        output_str = 'Epoch: {0:04d}/{1:04d} ({3:4.2f} s) loss train avg: {2:s}'.format(epoch+1, n_epochs, losses_avg_str, end-start)
                else:
                    if len(self.loss_functions) == 1:
                        metric_names = [metric.get_name() for task_id in range(len(self.loss_functions)) for metric in self.metrics[task_id]]
                    else:
                        metric_names = ['o{:d}_'.format(task_id) + metric.get_name() for task_id in range(len(self.loss_functions)) for metric in self.metrics[task_id]]
                    metric_names_display = '/'.join(metric_names)
                    metric_values_display = '/'.join(['{:4.2f}'.format(v) for v in measures_avg])
                    if loader_valid is not None:
                        losses_avg_str_v = '/'.join(['{:6.4f}'.format(loss_avg) for loss_avg in losses_valid_avg])
                        metric_values_v_display = '/'.join(['{:4.2f}'.format(v) for v in measures_valid_avg])
                        output_str = 'Epoch: {0:04d}/{1:04d}{6:1s} ({8:4.2f} s) loss train avg: {2:s}   {7:s} train avg: {3:s}   loss valid: {4:s}   {7:s} valid: {5:s}'.format(epoch+1, n_epochs, losses_avg_str, metric_values_display, losses_avg_str_v, metric_values_v_display, best_model_flag, metric_names_display, end-start)
                    else:
                        output_str = 'Epoch: {0:04d}/{1:04d} ({5:4.2f} s) loss train avg: {2:s}   {4:s} train avg: {3:s}'.format(epoch+1, n_epochs, losses_avg_str, metric_values_display, metric_names_display, end-start)  
                print(output_str, flush=True)
                
        return self
            
    # unaggregated prediction
    def predict_unagg(self, loader):
        
        # to evaluation mode
        self.network.eval()

        preds_raw = [None] * len(self.loss_functions)
        tgt = [None] * len(self.loss_functions)
            
        for step, batch in enumerate(loader):
            x, y, _ = self.prepare_batch(batch)
            
            # prediction
            outputs = singleton_to_list(self.network(*x))
            
            # gathering preds and targets            
            for task_id in range(len(self.loss_functions)):
                if preds_raw[task_id] is None:
                    preds_raw[task_id] = outputs[task_id].to(device='cpu')
                    tgt[task_id] = y[task_id].to(device='cpu')
                else:
                    preds_raw[task_id] = torch.cat((preds_raw[task_id], outputs[task_id].to(device='cpu')))
                    tgt[task_id] = torch.cat((tgt[task_id], y[task_id].to(device='cpu')))
                    
            #break

        # apply output function
        predictions = list(preds_raw)
        for task_id in range(len(self.loss_functions)):
            if self.output_fn[task_id] is not None:
                predictions[task_id] = self.output_fn[task_id](predictions[task_id])
        
        return predictions, preds_raw, tgt

    # helper function for aggregate predictions
    def aggregate_preds(self, preds, tgt):
        if self.pred_agg is None:
            return preds, tgt
        
        # more repeats for validation
        if self.n_repeats > 1:
            n_reps = self.n_repeats
            y_pred, y = [], []
            for p, t in zip(preds, tgt):
                pred_list = [p[torch.arange(n_reps,p.size(0)+n_reps,n_reps)-r-1] for r in range(n_reps)]
                new_p = pred_list[0]
                new_t = t[torch.arange(n_reps,p.size(0)+n_reps,n_reps)-1]
                for r in range(1, n_reps):
                    if self.pred_agg == 'max':
                        new_p = torch.maximum(new_p, pred_list[r])
                    elif self.pred_agg == 'average':
                        new_p += pred_list[r]
                if self.pred_agg == 'average':
                    new_p /= self.n_repeats
                y_pred.append(new_p)
                y.append(new_t)
        else:
            y_pred = preds
            y = tgt
        return y_pred, y
    
    
    def predict_and_eval(self, loader):
        # predict
        with torch.no_grad():
            y_pred, y_pred_raw, tgt = self.predict_unagg(loader)
        
        # aggregate
        y_pred_agg, tgt_agg = self.aggregate_preds(y_pred, tgt) 
        
        prediction_to_return = unpack_singleton([x.cpu().data.numpy() for x in y_pred_agg])

        # losses
        losses = [self.loss_functions[task_id](y_pred_raw[task_id], tgt[task_id]).item() for task_id in range(len(self.loss_functions))]
        if self.metrics is None:
            return prediction_to_return, losses, None
        
        # metrics
        measures_avg = [metric.compute(y_pred_agg[task_id], tgt_agg[task_id]) for task_id in range(len(self.loss_functions)) for metric in self.metrics[task_id]]
        return prediction_to_return, losses, measures_avg


    # prediction
    def predict(self, loader):
        # predict
        y_pred, _, _ = self.predict_and_eval(loader)        
        return y_pred
    
    
    # evaluation
    def eval(self, loader):
        # predict
        _, losses, measures_avg = self.predict_and_eval(loader)        
        return losses, measures_avg
    
    # load weights
    def load_weights(self, model_file, strict=False):
        state_dict = torch.load(model_file)
        state_dict = remove_prefix_from_state_dict_if_exists(state_dict, 'module.')
        self.network.load_state_dict(state_dict, strict)

    def save_weights(self, model_file):
        torch.save(self.network.state_dict(), f=model_file)
        
    def summary(self, *input_shape):
        device = next(self.network.parameters()).device
        data = [torch.rand(ishape, device=device) for ishape in input_shape]
        summary(self.network, *data, show_hierarchical=True, 
                print_summary=True, show_parent_layers=True, max_depth=None)