#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:39:37 2020

Report: fix-point quantization and dynamic range

@author: bhossein
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:18:34 2020

Training aware quantization


@author: bhossein
"""

# import time
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
#from torch.optim.lr_scheduler import _LRScheduler
#from torch.utils.data import TensorDataset, DataLoader
#import datetime
#import pickle
#from git import Repo

import torch.quantization
from torch.quantization import QuantStub, DeQuantStub

from torchsummary import summary

import pickle

import os
#abspath = os.path.abspath('test_classifier_GPU_load.py')
#dname = os.path.dirname(abspath)
#os.chdir(dname)

# os.chdir('/home/bhossein/BMBF project/code_repo')
#os.chdir('C:\Hinkelstien\code_repo')
res_dr = 'results/'

from my_data_classes import create_datasets_file, create_loaders, smooth, create_datasets_win
import my_net_classes
from my_net_classes import SepConv1d, _SepConv1d, Flatten, parameters
import torch
# import pickle
#import console
# %% ================ loading data
##load_ECG =  torch.load ('raw_x_8K_sync_win2K.pt')
#
#
save_name = "1d_4c_2fc_sub2_qr"
#
#
#raw_x = load_ECG['raw_x']
#target = load_ECG['target']
#data_tag = load_ECG['data_tag']
#
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#if type(target) != 'torch.Tensor':
#    target = torch.tensor(load_ECG['target']).to(device)
#
loaded_vars = pickle.load(open(res_dr+"train_"+save_name+"_variables.p","rb"))
#
#params = loaded_vars['params']
#epoch = params.epoch
#seed = params.seed
#test_size = params.test_size
#np.random.seed(seed)
#t_range = params.t_range
#
##dataset_splits = create_datasets_win(raw_x, target, data_tag, test_size, seed=seed, t_range = t_range, device = device)
#ecg_datasets = dataset_splits[0:3]
#trn_idx, val_idx, tst_idx = dataset_splits[3:6]
#trn_ds, val_ds, tst_ds = ecg_datasets
#
#batch_size = loaded_vars['params'].batch_size
batch_size = 1
#trn_dl, val_dl, tst_dl = create_loaders(ecg_datasets, bs=batch_size, jobs = 0)
#raw_feat = ecg_datasets[0][0][0].shape[0]
#raw_size = ecg_datasets[0][0][0].shape[1]
num_classes = 2

#%%===============  report

clear = lambda: os.system('clear') #on Windows System
clear()


#print ('''
#==============================================================================
#The summary of the 4-layer CNN network:      4Conv2FC
#==============================================================================
#       ''')
#model = pickle.load(open('train_'+save_name+'_best.pth', 'rb'))
#
#
#    
#raw_feat, raw_size = 2, 2048
#    
#print(model)
#
#
#
#print ('''
#       
#
#       
#==============================================================================
#Table of the network parameters:      4Conv2FC
#==============================================================================
#       ''')
#    
#model = model.to('cpu')
#summary(model, input_size=(raw_feat, raw_size), batch_size = batch_size, device = 'cpu')
#
##assert 1==2
#print ('''
#==============================================================================
#Detail of the network's per layer computations and parameters:      4Conv2FC
#==============================================================================
#       ''')
#
#flops, params = get_model_complexity_info(model, (raw_feat, raw_size), print_per_layer_stat=True, units = 'MMac')



#print ('''
#==============================================================================
#The network's accuracy:
#==============================================================================
#
#The network has the following accuracy:
#TP: 99.10 , FP: 3.88 AF_threshold = 3 (observing at least 3 AF signs to classify ECG as AF)
#TP: 98.77 , FP: 1.5 AF_threshold  = 7
#       ''')

#%
def model_summary(save_file, name1, name2):
    print ('''
    =======================================================================
    The summary of the 4-layer CNN network '''+name1+name2+
    '''
    =======================================================================
           ''')
    
    model = pickle.load(open(res_dr+save_file+'.pth', 'rb'))
    # model = torch.load(res_dr+save_file+'.pth', map_location = torch.device('cpu'))
          
    print(model)
    
    
    
    print ('''
           
    
           
    =======================================================================
    Table of the network parameters:'''+name2+'''
    =======================================================================
           ''')
        
    model = model.to('cpu')
    summary(model, input_size=(raw_feat, raw_size), batch_size = batch_size, device = 'cpu')
    
    #assert 1==2
    print ('''
    ================================================================================
    Detail of the network's per layer computations and parameters:'''+name2+'''
    ================================================================================
           ''')
    
    flops, params = get_model_complexity_info(model, (raw_feat, raw_size), print_per_layer_stat=True, units = 'MMac')
#%%
save_file = 'train_'+save_name+'_best'
name1 = ""
name2 = "      4Conv2FC"
model_summary(save_file, name1, name2)

#%%
name1 = "with conv. pruning:"
name2 = '''
     model: 4Conv2FC-conv-pruned    Conv-channels: 706 to 421'''
filter_per_iter = 1
thresh_AF = 7
epch_tr = 20
suffix = ""
#FC = "FC_200"
#FC = "_FC_100"
FC = ""
#suffix = "_bacc"
#save_name_pr = "prunned_"+save_name+"_"+str(filter_per_iter)+"fPi"
save_name_pr = "prunned_"+save_name+"_"+str(filter_per_iter)+"fPi_"+str(epch_tr)+"tpoch"+suffix
iteration = 200
save_file = save_name_pr+FC+"_iter_"+str(iteration)
model_summary(save_file, name1, name2)

#%%
name1 = "with conv. pruning:"
name2 = '''
     model: 4Conv2FC-conv-pruned    Conv-channels: 706 to 226'''
filter_per_iter = 1
thresh_AF = 7
epch_tr = 20
suffix = ""
FC = ""
save_name_pr = "prunned_"+save_name+"_"+str(filter_per_iter)+"fPi_"+str(epch_tr)+"tpoch"+suffix
iteration = 346
save_file = save_name_pr+FC+"_iter_"+str(iteration)
model_summary(save_file, name1, name2)
#%%  FC
name1 = "with FC. pruning:"
name2 = '''
     model: 4Conv2FC-FC-pruned    FC-nourons: 128 to 89'''
filter_per_iter = 1
thresh_AF = 7
epch_tr = 20
suffix = ""
FC = "FC_200"
save_name_pr = "prunned_"+save_name+"_"+str(filter_per_iter)+"fPi_"+str(epch_tr)+"tpoch"+suffix
iteration = 40
save_file = save_name_pr+FC+"_iter_"+str(iteration)
model_summary(save_file, name1, name2)

name1 = "with FC. pruning:"
name2 = '''
     model: 4Conv2FC-FC-pruned    FC-nourons: 128 to 42'''
filter_per_iter = 1
thresh_AF = 7
epch_tr = 20
suffix = ""
FC = "FC_200"
save_name_pr = "prunned_"+save_name+"_"+str(filter_per_iter)+"fPi_"+str(epch_tr)+"tpoch"+suffix
iteration = 85
save_file = save_name_pr+FC+"_iter_"+str(iteration)
model_summary(save_file, name1, name2)