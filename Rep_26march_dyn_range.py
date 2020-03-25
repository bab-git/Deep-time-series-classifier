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
#os.chdir('C:\Users\bhossein\Desktop\Hinkelsn\code_repo')
data_dr = 'data/'
res_dr = 'results/'

from my_data_classes import create_datasets_file, create_loaders, smooth, create_datasets_win
import my_net_classes
from my_net_classes import SepConv1d, _SepConv1d, Flatten, parameters
import torch
import pickle
#import console
initite = 0
# %% ================ loading data
if 'load_ECG' in locals() and initite == 0:
    print('''
        ==================      
          Using already extracted data
          ''')
    time.sleep(5)
else:    
#    load_ECG =  torch.load (data_dr+'raw_x_8K_sync_win2K.pt')
    load_ECG =  torch.load (data_dr+'raw_x_8K_nofilter_stable.pt')


#save_name ="2d_6CN_3FC_no_BN_in_FC_long"
save_name ="1d_flex_net_raw_data_8k_2c8_1fc8_pool4_6K"


raw_x = load_ECG['raw_x']
target = load_ECG['target']
data_tag = load_ECG['data_tag']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if type(target) != 'torch.Tensor':
    target = torch.tensor(load_ECG['target']).to(device)

loaded_vars = pickle.load(open(res_dr+"train_"+save_name+"_variables.p","rb"))

params = loaded_vars['params']
epoch = params.epoch
seed = params.seed
test_size = params.test_size
np.random.seed(seed)
t_range = params.t_range

if 'dataset_splits' not in locals():
    dataset_splits = create_datasets_win(raw_x, target, data_tag, test_size, seed=seed, t_range = t_range, device = device)

ecg_datasets = dataset_splits[0:3]
trn_idx, val_idx, tst_idx = dataset_splits[3:6]
trn_ds, val_ds, tst_ds = ecg_datasets

#batch_size = loaded_vars['params'].batch_size
batch_size = 1
trn_dl, val_dl, tst_dl = create_loaders(ecg_datasets, bs=batch_size, jobs = 0)
raw_feat = ecg_datasets[0][0][0].shape[0]
raw_size = ecg_datasets[0][0][0].shape[1]
num_classes = 2

#%%===============  loading a learned model

#save_name ="2d_6CN_3FC_no_BN_in_FC_long"

model = pickle.load(open(res_dr+'train_'+save_name+'_best.pth', 'rb'))
model.eval()
#%%===============  report
#!CLS
#%clear
#clear = lambda: os.system('clear') #on Windows System
#clear()


print ('''
==============================================================================
The summary of the 2-layer CNN network:
==============================================================================
       ''')
    
raw_feat, raw_size = 2, 6016
    
print(model)



print ('''
       

       
==============================================================================
Table of the network parameters:         
==============================================================================
       ''')
    
model = model.to('cpu')    
summary(model, input_size=(raw_feat, raw_size), batch_size = 1, device = 'cpu')

#assert 1==2
print ('''
==============================================================================
Detail of the network's per layer computations and parameters:
==============================================================================
       ''')

flops, params = get_model_complexity_info(model, (raw_feat, raw_size), print_per_layer_stat=True, units = 'MMac')


#
#print ('''
#==============================================================================
#The network's accuracy:
#==============================================================================
#
#The network has the following accuracy:
#TP: 99.10 , FP: 3.88 AF_threshold = 3 (observing at least 3 AF signs to classify ECG as AF)
#TP: 98.77 , FP: 1.5 AF_threshold  = 7
#       ''')



# %%-------------- stats on weights

print ('''
==============================================================================
Statistical analysis of the weights:
==============================================================================        
       ''')    


def stat_analysis(params,title, weight = 'weights', color = 'g'):
    
    print ('Minimum of '+weight+': %3.7f' %(params.min()))
    print ('Maximum of '+weight+': %3.7f' %(params.max()))
    print ('Average value of '+weight+': %3.7f' %(params.mean()))
    print ('Variance of '+weight+': %3.7f' %(np.std(np.array(params))))
    
    plt.figure(figsize = (25,5))
    n, bins, patches = plt.hist(params, 30, facecolor = color , rwidth = 0.75)
    if max(bins) <1000 and min(bins) >-1000:
#        xbins = np.floor(bins)
#    else:
        xbins = np.floor(bins*10)/10
        plt.xticks(bins,xbins)
    plt.xlabel('Values')
    plt.ylabel('Counts')
    plt.title(title)
    plt.grid(True)
    plt.show()

    
params = [];
model = model.to('cpu')
for m in model.modules():
#    print(m)
    if type(m) == nn.Conv2d:
#        print(m)
        params = np.append(params,np.array(m.weight.data.view(-1)))
#        np.append(params, m.weight.data)
    
    elif type(m) == nn.Linear:
#        print(m)        
        params = np.append(params,np.array(m.weight.data.view(-1)))
        
    elif type(m) == nn.BatchNorm2d:
#        print(m)        
        params = np.append(params,np.array(m.weight.data.view(-1)))
        



#print ('Minimum of absolute weights: %3.7f' %(np.abs(params).min()))

print('''
         
============= All weights of the network
          ''')
stat_analysis(params,'Histogram of all weights')

for i in range(1,3):
    params = []
    for j in range(3):
        m = model.raw[i].layers[j]
#        params = m.weight.data.view(-1)
        params = np.append(params,np.array(m.weight.data.view(-1)))
#    if type(m)     == 
    print('''

          
============= Conv. Layer: %d '''
           %(i+1))
    stat_analysis(params,'Histogram of weights in Conv-layer: %d' %(i+1))

c = 0
i =1    
c += 1
m = model.FC[i].to('cpu')
params = np.array(m.weight.data.view(-1))
print('''
          
============= Fully connected Layer: %d '''
           %(c))
stat_analysis(params,'Histogram of weights in FC-layer: %d' %(c))
        

# %%-------------- stats on feature maps
print ('''


       

       

==============================================================================
Statistical analysis of the intermediate signals:
============================================================================== ''')    
    
model.eval()    
# Visualize feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

i_batch,batch = next(enumerate(tst_dl))
x_raw, y_target = [t.to(device) for t in batch]

model = model.to(device)
for i in range(1,3):
    params = []
#    for j in range(1,4):
    for j in range(0,4):        
#        m = model.raw[i].layers[j]        
        model.raw[i].layers[j].register_forward_hook(get_activation('hook %d_%d'%(i,j)))
        modelb = model.raw[i].layers[:j+1]
        if i>0:
            modela = model.raw[:i]
            modelb = nn.Sequential(modela,modelb)
        
        y = modelb(x_raw.unsqueeze(2))
#        model(x_raw)
#        act = activation['hook %d_%d'%(i,j)]
#        features = act.view(-1).to('cpu')
        features = y.data.view(-1).to('cpu')
        if j==0:
            module = 'Conv. 1'
        elif j==1:
            module = 'Conv. 2'
        elif j==2:
            module = 'batch normalization'
        else:
            module = 'ReLU'
            features = np.delete(features,np.where(features == 0))
            
        print('''
              
              ''')
        print('============= Feature-values, layer %d ,'%(i+1)+module+' output: ')
    
        stat_analysis(features ,"Histogram of feature-values, layer %d , "%(i+1)+module+' output', 'feature-values', color = 'b')

# %%
y_raw = model.raw(x_raw.unsqueeze(2))
for i in [1,4]:
    params = []
    for j in range(2):
#        m = model.raw[i].layers[j]
        
#        model.FC[i+j].register_forward_hook(get_activation('hook_fc %d'%(i+j)))
        modela = model.FC[:i+j+1]
        y = modela(y_raw)
        
#        model(x_raw)
#        act = activation['hook_fc %d'%(i+j)]
#        features = act.view(-1).to('cpu')
        features = y.data.view(-1).to('cpu')
        if j==0:
            module = 'Linear layer'                
        else:
            module = 'ReLU'
            features = np.delete(features,np.where(features == 0))
            
        print('''
              
              ''')
        print('============= Feature-values, FC layer %d ,'%(i)+module+' output: ')
    
        stat_analysis(features ,"Histogram of feature-values, FC layer %d , "%(i)+module+' output', 'feature-values', color = 'b')
    
    


#model.raw[0].layers[0].register_forward_hook(get_activation('conv1'))
#model.raw[0].layers[3].register_forward_hook(get_activation('relu1'))
#
#i_batch,batch = next(enumerate(tst_dl))
#x_raw, y_target = [t.to(device) for t in batch]
#
#out = model(x_raw)
#act = activation['conv1']
#act_relu = activation['relu1']
#
#features = act_relu.view(-1).to('cpu')
#
#stat_analysis(features ,'Histogram of feature-values in ??? : %d' %(c))

