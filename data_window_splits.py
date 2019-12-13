#import time
#from collections import defaultdict
#from functools import partial
#from multiprocessing import cpu_count
#from pathlib import Path
#from textwrap import dedent
#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd

#from sklearn.externals import joblib
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
#from torch import nn
#from torch import optim
#from torch.nn import functional as F
#from torch.optim.lr_scheduler import _LRScheduler
#from torch.utils.data import TensorDataset, DataLoader
#import datetime
#import pickle
#from git import Repo

import os
#abspath = os.path.abspath('test_classifier_GPU_load.py')
#dname = os.path.dirname(abspath)
#os.chdir(dname)

os.chdir('/home/bhossein/BMBF project/code_repo')
#os.chdir('C:\Hinkelstien\code_repo')

from my_data_classes import create_datasets, create_loaders, read_data, create_datasets_file, smooth
#import my_net_classes
#from my_net_classes import SepConv1d, _SepConv1d, Flatten, parameters

#%% =======================
#seed = 11
#seed = int(input ('Enter seed value for randomizing the splits (default = 11):'))
#np.random.seed(seed)

#==================== data IDs

#IDs = []
#main_path = '/vol/hinkelstn/data/FILTERED/atrial_fibrillation_8k/'    
#IDs.extend(os.listdir(main_path))
#IDs = os.listdir(main_path)
#main_path = '/vol/hinkelstn/data/FILTERED/sinus_rhythm_8k/'
#IDs.extend(os.listdir(main_path))
#
#target = np.ones(16000)
#target[0:8000]=0

#t_range = range(1000,1512)

t_win = 2**11
t_shift = 400
#t_shift = None

t_range = range(t_win)
#load_ECG =  torch.load ('raw_x_all.pt') 
#load_ECG =  torch.load ('raw_x_40k_50K.pt') 
#load_ECG =  torch.load ('raw_x_6K.pt') 
load_ECG =  torch.load ('raw_x_8K_sync.pt')
            

#%%==================== test and train splits
"creating dataset"     
#test_size = 0.25

#cuda_num = input("enter cuda number to use: ")

#device = torch.device('cuda:'+cuda_num if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


raw_x = load_ECG['raw_x']    
target = load_ECG['target']

length = raw_x.shape[2]
n_win = np.floor(length / t_shift) - np.floor(t_win / t_shift)    
raw_x_extend = torch.empty([raw_x.shape[0]*n_win.astype(int),raw_x.shape[1],t_win])
target_extend = np.zeros(raw_x.shape[0]*n_win.astype(int))    
data_tag = np.zeros(raw_x.shape[0]*n_win.astype(int))


print("Extracting temporal windows from ECG files..." )
                                  
for i_data in range(raw_x.shape[0]):    
    if i_data % 500 ==0:
        print("data number: "+str(i_data))
    for i_win in range(int(n_win)):
                
        i_ext = i_data*n_win+i_win
        
        raw_x_extend[int(i_ext),:,:] = raw_x[i_data,:,i_win*t_shift:i_win*t_shift+t_win] 
        target_extend[int(i_ext)] = target[i_data]
        data_tag[int(i_ext)] = i_data

del raw_x, target
raw_x = raw_x_extend
target = target_extend

#%%
torch.save({'raw_x':raw_x,'target':target,'data_tag':data_tag}, "raw_x_8K_sync_win2K.pt")
#torch.save({'raw_x':raw_x,'target':target,'target_extend':target_extend,'data_tag':data_tag}, "raw_x_8K_sync_win2K.pt")