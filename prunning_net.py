#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:24:54 2020

@author: bhossein
"""

import time
#from collections import defaultdict
#from functools import partial
#from multiprocessing import cpu_count
#from pathlib import Path
#from textwrap import dedent
import matplotlib.pyplot as plt
import numpy as np
import random
#import pandas as pd

#from sklearn.externals import joblib
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder, StandardScaler

from ptflops import get_model_complexity_info
from thop import profile

#import torch

from torch import nn
#from torch import optim
from torch.nn import functional as F
from torch.nn.utils import prune
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
#from torch.optim.lr_scheduler import _LRScheduler
#from torch.utils.data import TensorDataset, DataLoader

from torchsummary import summary

#import datetime

#from git import Repo

import copy

import option_utils

import pickle

import os
#abspath = os.path.abspath('test_classifier_GPU_load.py')
#dname = os.path.dirname(abspath)
#os.chdir(dname)

os.chdir('/home/bhossein/BMBF project/code_repo')
#os.chdir('C:\Hinkelstien\code_repo')

from my_data_classes import create_datasets_file, create_loaders, smooth, create_datasets_win
import my_net_classes
from my_net_classes import SepConv1d, _SepConv1d, Flatten, parameters
import torch
import pickle

from evaluation import evaluate

from  prunning_class import PrunningFineTuner

#%% ============== options
model_cls, model_name   = option_utils.show_model_chooser()
dataset, data_name  = option_utils.show_data_chooser()
save_name           = option_utils.find_save(model_name, data_name)
if save_name == 'NONE':
    save_name = "1d_4c_2fc_sub2_qr"
#    save_name ="2d_6CN_3FC_no_BN_in_FC_long"
    #save_name ="2d_6CN_3FC_no_BN_in_FC"
    #save_name = "test_full_6conv_long"
    #save_name = "test_6conv_v2"
    #save_name = "test_full_6conv"
    #save_name = "1d_6_conv_qn"
    #save_name = "1d_5conv_2FC_qn"
    #save_name = "1d_3conv_2FC_v2_seed2"
    #save_name = "1d_3conv_2FC_seed2"
    #save_name = "1d_3conv_2FC_v2_2K_win"
    #save_name = "1d_1_conv_1FC"
    #save_name = "1d_3con_2FC_2K_win"
    #save_name = "1d_6con_2K_win_2d"
    #save_name = "1d_6con_2K_win_test_30"
    #save_name = "1d_6con_b512_trim_2K_win"
    #save_name = "1d_6con_b512_trim_2K_win_s11"
    #save_name = "1d_6con_b512_trim_2K_win_s3"
    #save_name = "1d_6con_b512_trim_2K_seed2"
    #save_name = "1dconv_b512_t4K"
    #save_name = "1dconv_b512_drop1B"
    #save_name = "1dconv_b512_drop1"
    #save_name = "batch_512_BN_B"
    #save_name = "1dconv_b512_BNM_B"
    #save_name = "1dconv_b512_BNA_B"
    #save_name = "batch_512_BNA"
    #save_name = "batch_512_BN"
    #save_name = "batch_512_B"
    #t_stamp = "_batch_512_11_29_17_03"

print("{:>40}  {:<8s}".format("Selected experiment:", save_name))

#device              = option_utils.show_gpu_chooser(default=1)
cuda_num = 0   # export CUDA_VISIBLE_DEVICES=x
device = torch.device('cuda:'+str(cuda_num) if torch.cuda.is_available() and cuda_num != 'cpu' else 'cpu')
torch.cuda.device(cuda_num)
# %% ================ loading data
print("{:>40}  {:<8s}".format("Loading dataset:", dataset))

load_ECG = torch.load(dataset)

#%%===============  loading experiment's parameters and batches

print("{:>40}  {:<8s}".format("Loading model:", model_name))

loaded_vars = pickle.load(open("train_"+save_name+"_variables.p","rb"))
#loaded_file = pickle.load(open("variables"+t_stamp+".p","rb"))
#loaded_file = pickle.load(open("variables_ended"+t_stamp+".p","rb"))

params = loaded_vars['params']
epoch = params.epoch
print("{:>40}  {:<8d}".format("Epoch:", epoch))
seed = params.seed
print("{:>40}  {:<8d}".format("Seed:", seed))
test_size = params.test_size
np.random.seed(seed)
t_range = params.t_range

#cuda_num = 0   # export CUDA_VISIBLE_DEVICES=x

#device = torch.device('cuda:'+str(cuda_num) if torch.cuda.is_available() and cuda_num != 'cpu' else 'cpu')
#device = torch.device('cpu')

raw_x = load_ECG['raw_x']
#raw_x = load_ECG['raw_x'].to(device)
target = load_ECG['target']
data_tag = load_ECG['data_tag']
#data_tag = load_ECG['IDs']
if type(target) != 'torch.Tensor':
    target = torch.tensor(load_ECG['target']).to(device)

dataset_splits = create_datasets_win(raw_x, target, data_tag, test_size, seed=seed, t_range = t_range, device = device)
ecg_datasets = dataset_splits[0:3]
trn_idx, val_idx, tst_idx = dataset_splits[3:6]
#ecg_datasets = create_datasets_file(raw_x, target, test_size, seed=seed, t_range = t_range, device = device)

acc_history = loaded_vars['acc_history']
loss_history = loaded_vars['loss_history']
#ecg_datasets = loaded_split['ecg_datasets']
trn_ds, val_ds, tst_ds = ecg_datasets

batch_size = loaded_vars['params'].batch_size
trn_dl, val_dl, tst_dl = create_loaders(ecg_datasets, bs=batch_size, jobs = 0)
raw_feat = ecg_datasets[0][0][0].shape[0]
raw_size = ecg_datasets[0][0][0].shape[1]
num_classes = 2

# %%   loading the pre-trained model

model = model_cls(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
model0 = copy.deepcopy(model)

try:
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("train_"+save_name+'_best.pth', map_location=lambda storage, loc: storage.cuda(device)))
    else:
        model.load_state_dict(torch.load("train_"+save_name+'_best.pth', map_location=lambda storage, loc: storage))
except:    
    model = pickle.load(open('train_'+save_name+'_best.pth', 'rb'))

thresh_AF = 5

TP_ECG_rate, FP_ECG_rate, list_pred_win, elapsed = evaluate(model, tst_dl, tst_idx, data_tag, thresh_AF = thresh_AF, device = device)

# %% ================== Pruning
model = model_cls(raw_feat, num_classes, raw_size, batch_norm = True).to(device)


from  prunning_class import PrunningFineTuner
print('''class loaded 
      
''')

fine_tuner = PrunningFineTuner(trn_dl, val_dl, model)

prune_targets = fine_tuner.prune()

layer_history = []
for layer_index, filter_index in prune_targets:    
    if layer_index not in layer_history:
        print("prunning layer:" , layer_index)
        layer_history.append(layer_index)
    model = prune_conv_layers(model, layer_index, filter_index)

model(x_raw).shape    
                
#layers_prunned = {}
#for layer_index, filter_index in prune_targets:
#    print(layer_index, filter_index)
#    
#    if layer_index not in layers_prunned:
#        layers_prunned[layer_index] = 0
#    layers_prunned[layer_index] = layers_prunned[layer_index] + 1 



# %%
#assert 1==1

i_batch, batch = next(enumerate(trn_dl))
x_raw, y_target = batch
x = x_raw
#for layer, module in enumerate(model.modules()):


#self = []
activations = []
gradients = []
grad_index = 0
activation_to_layer = {}

model.train()
layer = 1
x = x_raw.cpu().unsqueeze(2)
module = model.raw[0]
x = module(x)
for i_layer in range(1,5):  #4c2fc    
    for (name,module) in model.raw[i_layer].layers._modules.items():
        layer += 1
        print (layer," ",module)
        x = module(x)
        if type(module) == nn.Conv1d or type(module) == nn.Conv2d:
            assert 1==2
            x.register_hook(self.compute_rank)
#            print (i_layer," ",module)
            print (" ")
            
            activation_index = len(activations) - grad_index - 1
        	activation = activations[activation_index]
        	values = \
        		torch.sum((activation * grad), dim = 0).\
        			sum(dim=2).sum(dim=3)[0, :, 0, 0].data














x = torch.randn(1, 1)
w = torch.randn(1, 1, requires_grad=True)
w.register_hook(lambda x: print(x))
y = torch.randn(1, 1)

out = x * w
loss = (out - y)**2
loss.register_hook(lambda x: print(x))
#loss.mean().backward(gradient=torch.tensor([0.1]))  # prints the gradient in w and loss
loss.mean().backward()  # prints the gradient in w and loss
print("")


#model.eval()
#y1 = model.raw(x_raw.cpu().unsqueeze(2))
#y2 = x
#torch.norm(y1.data-y2.data, p=2)


#model.out    
#    print ()
#    x = module(x)
#    if type(module) == nn.Conv1d or type(module) == nn.Conv2d:
#        print (layer,"  ", name,"  ", module)
#        print (" ")
            
 
#utput = self.model(Variable(batch))
#            pred = output.data.max(1)[1]
#            correct += pred.cpu().eq(label).sum()
#            total += label.size(0)
#            
#            
#total = 0
#correct = 0
#for i_batch, batch in enumerate(tst_dl):
#    x_raw, y_batch = [t.to(device) for t in batch]
##    list_x = list(range(i_batch*tst_dl.batch_size,min((i_batch+1)*tst_dl.batch_size,len(tst_dl.dataset))))
#    # x_raw, y_batch = [t.to(device) for t in batch]
#    # x_raw, y_batch = tst_ds.tensors
#    # x_raw, y_batch = [t.to(device) for t in val_ds.tensors]
##    out = model(x_raw)
#    out = model(Variable(x_raw))
#    preds = out.argmax(dim=1)
#    # preds = torch.sigmoid(out).squeeze().round()
#    # preds = F.log_softmax(out, dim = 1).argmax(dim=1)
#    # preds = F.log_softmax(out, dim = 1).argmax(dim=1).to('cpu')
##    list_pred = np.append(list_pred,preds.cpu())
#    # list_pred = np.append(list_pred,preds.tolist())
#    total += y_batch.size(0)
#    correct += (preds ==y_batch).sum().item()    
#    # i_error = np.append(i_error,np.where(preds !=y_batch))
##    i_error = np.append(i_error,[list_x[i] for i in np.where((preds !=y_batch).cpu())[0]])
#    # TP += ((preds ==y_batch) & (1 ==y_batch)).sum().item()
#    # total_P += (1 ==y_batch).sum().item()
#    # FP += ((preds !=y_batch) & (0 ==y_batch)).sum().item()
#print("Accuracy :", float(correct) / total)