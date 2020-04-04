#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 16:39:00 2020

@author: bhossein
"""

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
    load_ECG =  torch.load (data_dr+'raw_x_8K_sync_win2K.pt')


save_name ="2d_6CN_3FC_no_BN_in_FC_long"


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

save_name ="2d_6CN_3FC_no_BN_in_FC_long"

model = pickle.load(open(res_dr+'train_'+save_name+'_best.pth', 'rb'))

#%%===============  report

clear = lambda: os.system('clear') #on Windows System
clear()


print ('''
==============================================================================
The summary of the 6-layer CNN network:
==============================================================================
       ''')
    
raw_feat, raw_size = 2, 2048
    
print(model)



print ('''
       

       
==============================================================================
Table of the network parameters:         
==============================================================================
       ''')
    
model = model.to('cpu')    
summary(model, input_size=(raw_feat, raw_size), batch_size = batch_size, device = 'cpu')

#assert 1==2
print ('''
==============================================================================
Detail of the network's per layer computations and parameters:
==============================================================================
       ''')

flops, params = get_model_complexity_info(model, (raw_feat, raw_size), print_per_layer_stat=True, units = 'MMac')



print ('''
==============================================================================
The network's accuracy:
==============================================================================

The network has the following accuracy:
TP: 99.10 , FP: 3.88 AF_threshold = 3 (observing at least 3 AF signs to classify ECG as AF)
TP: 98.77 , FP: 1.5 AF_threshold  = 7
       ''')



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

for i in range(6):
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
for i in [1,4]:    
    c += 1
    m = model.FC[i].to('cpu')
    params = np.array(m.weight.data.view(-1))
    print('''
          
============= Fully connected Layer: %d '''
           %(c))
    stat_analysis(params,'Histogram of weights in FC-layer: %d' %(c))
        

# %%-------------- stats on feature maps
%matplotlib inline    
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

#i_batch,batch = next(enumerate(tst_dl))
#x_raw, y_target = [t.to(device) for t in batch]
x = trn_ds.tensors[0]
x = x.unsqueeze(2)

print('============= Feature-values, Input: ')

features = x.data.view(-1).to('cpu')        
stat_analysis(features ,"Histogram of feature-values, input layer", 'feature-values', color = 'b')           

model = model.to(device)
for i_layer in range(3):              #raw layers: 
#            print(i_layer)
    module = model.raw[i_layer]
    if type(model.raw[i_layer]) == nn.MaxPool2d:
        x = module(x)
        continue
    for (name,module) in model.raw[i_layer].layers._modules.items():
        x = module(x)
        if type(module) in (nn.Conv2d, nn.BatchNorm2d, nn.ReLU):
            features = x.data.view(-1).to('cpu')
            if type(module) == nn.ReLU:
                features = np.delete(features,np.where(features == 0))
            
            module_n = type(module)

            print('''
                  
                  ''')
            print('============= Feature-values, layer %d ,'%(i_layer+1)+str(module_n)+' output: ')
        
            stat_analysis(features ,"Histogram of feature-values, layer %d , "
                          %(i_layer+1)+str(module_n)+' output', 'feature-values', color = 'b')           

for (name,module) in model.FC._modules.items():
    x = module(x)
    if type(module) in (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Linear):
        features = x.data.view(-1).to('cpu')
        if type(module) == nn.ReLU:
            features = np.delete(features,np.where(features == 0))
        
        module_n = type(module)

        print('''
              
              ''')
        print('============= Feature-values, FC layer %d ,'+str(module_n)+' output: ')
    
        stat_analysis(features ,"Histogram of feature-values, layer %d , "
                      %(i_layer+1)+str(module_n)+' output', 'feature-values', color = 'b')           
