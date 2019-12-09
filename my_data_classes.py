#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:12:40 2019

@author: bhossein
"""

#import time
#from collections import defaultdict
#from functools import partial
#from multiprocessing import cpu_count
#from pathlib import Path
#from textwrap import dedent
#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import pickle

import time

#from sklearn.externals import joblib
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
#from torch import nn
#from torch import optim
#from torch.nn import functional as F
#from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader

from torch.utils import data
#from torch.utils.data import TensorDataset

from sklearn.model_selection import train_test_split

import wavio

import matplotlib.pyplot as plt

from scipy import stats
import scipy.stats

import os
# %%==============    

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'    
    def __init__(self,list_IDs, labels, t_range=None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs        
        self.t_range = t_range
#        self.path = path
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        y = self.labels[index]
        assert y <= self.labels.max()
        # Load data and get label
        if y == 0:
            main_path = '/vol/hinkelstn/data/FILTERED/atrial_fibrillation_8k/'
#            main_path = '/data/bhosseini/hinkelstn/FILTERED/atrial_fibrillation_8k/'
        else:
            main_path = '/vol/hinkelstn/data/FILTERED/sinus_rhythm_8k/'
#            main_path = '/data/bhosseini/hinkelstn/FILTERED/sinus_rhythm_8k/'
            
            
        
            
#        list_f = os.listdir(main_path)        
        path = main_path+ID
        w = wavio.read(path)
        w_zm = stats.zscore(w.data,axis = 0, ddof = 1)
#        X = w.data.transpose(1,0)
        if self.t_range:
            X = torch.tensor(w_zm[self.t_range,:].transpose(1,0)).float()
        else:
            X = torch.tensor(w_zm.transpose(1,0)).float()
                
#        X = torch.tensor(w.data.transpose(1,0)).view(1,2,X.shape[1])
        
        
        y = torch.tensor(y).long()
#        y = torch.tensor(y).view(1,1,1)
                  
#        data_tensor = TensorDataset(X.float(),y.long())
        
        return X, y
        
#%% ================== PyTorch Datasets and Data Loaders
def create_datasets(IDs, target, test_size, valid_pct=0.1, seed=None, t_range=None):
    """
    Creating train/test/validation splits
    
    Three datasets are created in total:
        * training dataset
        * validation dataset
        * testing dataset
    """
    
    idx = np.arange(len(IDs))
    trn_idx, tst_idx = train_test_split(idx, test_size=test_size, random_state=seed)
    val_idx, tst_idx= train_test_split(tst_idx, test_size=0.5, random_state=seed)
    
    
    
    
    trn_ds = Dataset([IDs[i] for i in trn_idx],target[trn_idx],t_range)
    tst_ds = Dataset([IDs[i] for i in tst_idx],target[tst_idx],t_range)
    val_ds = Dataset([IDs[i] for i in val_idx],target[val_idx],t_range)
    
    return trn_ds, val_ds, tst_ds

#%% ================== 
def create_loaders(data, bs=128, jobs=0):
    """Wraps the datasets returned by create_datasets function with data loaders."""
    
    trn_ds, val_ds, tst_ds = data
    
    if bs == "full":
        bs_trn = len(trn_ds)
        bs_tst = len(tst_ds)
        bs_val = len(val_ds)
    else:
        bs_trn = bs_tst = bs_val = bs
        
    trn_dl = DataLoader(trn_ds, batch_size=bs_trn, shuffle=True, num_workers=jobs)
    val_dl = DataLoader(val_ds, batch_size=bs_tst, shuffle=False, num_workers=jobs)
    tst_dl = DataLoader(tst_ds, batch_size=bs_val, shuffle=False, num_workers=jobs)
    return trn_dl, val_dl, tst_dl             

#%% ================== read all data 
def read_data(save_file = 'temp_save' , t_length = 7500 , t_base = 3000, t_range = None):
    IDs = []
#    main_path = '/vol/hinkelstn/data/FILTERED/atrial_fibrillation_8k/'
    main_path = 'C:\Hinkelstien\data/FILTERED/atrial_fibrillation_8k/'
    IDs.extend(os.listdir(main_path))
    IDs = os.listdir(main_path)
#    main_path = '/vol/hinkelstn/data/FILTERED/sinus_rhythm_8k/'
    main_path = 'C:\Hinkelstien\data/FILTERED/sinus_rhythm_8k/'
    IDs.extend(os.listdir(main_path))

    target = np.ones(16000)
    target[0:8000]=0
#    t_range = range(0,6000)
#    t_range = range(40000,50000)
#    t_range = range(0,60000)
    
    raw_x=torch.empty((len(IDs),2,t_length), dtype=float)
#    raw_x=torch.empty((len(IDs),2,len(t_range)), dtype=float)
    i_ID=0;
    #    for i, ID in enumerate(IDs):
#    list_reject = []    
    millis = (time.time())
    millis2 = (time.time())
    t_start = 0;  T = 0
    while i_ID< len(IDs):        
        del t_start, T
        
        ID = IDs[i_ID]
        print('sample: %d , time: %5.2f (s)' % (i_ID, millis2-millis))
        
        millis = (time.time())
#        pickle.dump({'i_ID':i_ID},open("read_data_i_ID.p","wb"))
        if i_ID % 100 == 0:
#            pickle.dump({'i_ID':i_ID},open("read_data_i_ID.p","wb"))
            print(i_ID)
        y = target[i_ID]
        assert y <= target.max()
        # Load data and get label
        if y == 0:
#            main_path = '/vol/hinkelstn/data/FILTERED/atrial_fibrillation_8k/'
            main_path = 'C:\Hinkelstien\data/FILTERED/atrial_fibrillation_8k/'
        else:
#            main_path = '/vol/hinkelstn/data/FILTERED/sinus_rhythm_8k/'
            main_path = 'C:\Hinkelstien\data/FILTERED/sinus_rhythm_8k/'            
        path = main_path+ID
        w = wavio.read(path)
        
        #------------------ cliping the range
#        reject_flag = 0
        T = len(w.data)
        data_trim=np.zeros([t_length,w.data.shape[1]])
                
        data = w.data
        t_base = 3000
        max_list = [data[i*t_base:(i+1)*t_base].max(axis = 0) for i in range(0,np.floor(T/t_base).astype(int))]
        mean_max = np.mean(max_list, axis =0)    
        list_peaks = np.where(np.sum(data > mean_max, axis = 1))[0]
        list_p1 = np.roll(list_peaks,1)
        list_p1[0] = 0
        del_p = (list_peaks-list_p1)
        list_p2 = [list_peaks[i] for i in np.where(del_p>1)[0]]
        crop_t = np.unique([i for t in list_p2 for i in range(t-400,t+400)])
        crop_t = np.delete(crop_t,np.where((crop_t < 0) | (crop_t > T)))    
#        crop_t = np.random.randint(1,1000,1000)
        
        trimmed_t = np.arange(T)
        trimmed_t = np.delete(trimmed_t,crop_t)
        
#        trimmed_t = [i for i in range(T) if i not in crop_t]
#        trimmed_t = range(T)
       
#        trimm_out = wave_harsh_peaks_all(w.data, t_base = 3000)        
#        mean_max, trimmed_t = trimm_out            
#        trimmed_t = list(range(T))
        
        
        list_t0 = trimmed_t-np.roll(trimmed_t,1)
        list_t = np.where(list_t0 !=1)[0]
        list_t = np.append(list_t,len(trimmed_t))
        if len(list_t)==1:
            t_start = 1
        else:
            
            temp1 = list_t-np.roll(list_t,1)
            ind_list = np.where(t_length+1 < temp1)[0]
            assert len(ind_list) > 0
            t_start = list_t[ind_list[0]-1]+1
               
##-------------- loop version     
        reject_flag = 0
        list_t = trimmed_t-np.roll(trimmed_t,1)
        list_t[(list_t != 1) & (list_t != 0)] = 0
        for i_t in range(len(list_t)):
            if i_t+t_length > w.data.shape[0]:
#                list_reject = np.append(list_reject,ID)
                reject_flag = 1
                break
            if sum(list_t[i_t:i_t+t_length]) == t_length:
                break
        assert reject_flag == 0
        
        t_start0 = i_t
        
        assert t_start0 == t_start
##-------------- loop version             

        
        t_select = trimmed_t[t_start:t_start+t_length]
        data_trim = w.data[t_select,:]
        
#        plt.figure()
#        for i_f in range(2):
#            plt.subplot(2,1,i_f+1)
#            plt.plot(w.data[:,i_f], color = 'b')
##                plt.scatter(list_peaks, w.data[list_peaks.astype(int),i_f], color = 'g')    
##            plt.scatter(t_select, data_trim[:,i_f], color = 'y')
#            plt.scatter(trimmed_t, w.data[trimmed_t,i_f], color = 'y')
##                plt.scatter(crop_t, data[crop_t.astype(int),i_f], color = 'g')         
##                
        w.data = data_trim
        w_zm = stats.zscore(w.data,axis = 0, ddof = 1)
        if t_range:
            X = torch.tensor(w_zm[t_range,:].transpose(1,0)).float()
        else:
            X = torch.tensor(w_zm.transpose(1,0)).float()
        
        raw_x[i_ID,:,:]= X
        i_ID +=1
        
        millis2 = (time.time())
        
        #        X = torch.tensor(w.data.transpose(1,0)).view(1,2,X.shape[1])     
    torch.save({'raw_x':raw_x, 'target':target}, save_file+'.pt')
    return target, raw_x


#plt.figure()
#plt.subplot(211)
#plt.plot(w.data[:,0])
#plt.subplot(212)
#plt.plot(w.data[:,1])
#
#data_trim = raw_x[i_ID-2,:,:]
#plt.figure()
#plt.subplot(211)
#plt.plot(data_trim[0,:])
#plt.subplot(212)
#plt.plot(data_trim[1,:])
#%% ================== test/train using all read data
def create_datasets_file(raw_x, target, test_size, valid_pct=0.1, seed=None, t_range=None):
    """
    Creating train/test/validation splits
    
    Three datasets are created in total:
        * training dataset
        * validation dataset
        * testing dataset
    """
#    raw_x = torch.load ('raw_x_all.pt') 
    
#    raw_t = raw_x[trn_idx,:,t_range.start:t_range.stop]
       
    idx = np.arange(raw_x.shape[0])
#    idx = raw_x.shape[0]
    trn_idx, tst_idx = train_test_split(idx, test_size=test_size, random_state=seed)
    val_idx, tst_idx= train_test_split(tst_idx, test_size=0.5, random_state=seed)
    
    
    trn_ds = TensorDataset(raw_x[trn_idx,:,t_range.start:t_range.stop].float(),
                           target[trn_idx].long())
    tst_ds = TensorDataset(raw_x[tst_idx,:,t_range.start:t_range.stop].float(),
                           target[tst_idx].long())
    val_ds = TensorDataset(raw_x[val_idx,:,t_range.start:t_range.stop].float(),
                           target[val_idx].long())
    
    return trn_ds, val_ds, tst_ds

#%% ================== smoothening the output
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode = 'same')
    return y_smooth


#%% ================== smoothening the output
def wave_harsh_peaks(data, th_ratio = 3, ax  = None, t_base = 3000):
    T = len(data)    
    
#    for i in range(0,np.floor(T/t_base).astype(int)):
#        np.mean(data[i*t_base:(i+1)*t_base])
    max_list = [max(data[i*t_base:(i+1)*t_base]) for i in range(0,np.floor(T/t_base).astype(int))]
    mean_max = np.mean(max_list)
    thresh = mean_max*th_ratio

    list_p = np.where(data>mean_max)[0]    
    list_p1 = np.roll(list_p,1)
    list_p1[0] = 0
    del_p = (list_p-list_p1)
    list_p2 = [list_p[i] for i in np.where(del_p>1)[0]]
    
    crop_t = []
    for t in list_p2:
        crop_t = np.append(crop_t,np.arange(t-400,t))
        crop_t = np.append(crop_t,np.arange(t,t+400))
    
    crop_t = np.delete(crop_t,np.where((crop_t < 0) | (crop_t > len(data))))
    
    trimmed_t = [i for i in range(len(data)) if i not in crop_t]
#    plt.plot(trimmed_t, data[trimmed_t], color = 'y')
    
    if ax is not 'silent':
        if not ax:
           plt.figure()
           ax = plt
        
        ax.grid()
        ax.scatter(range(len(max_list)), max_list)
        ax.scatter(range(len(max_list)), mean_max*np.ones(len(max_list)), color = 'g')
        ax.scatter(range(len(max_list)), thresh*np.ones(len(max_list)), color = 'r')
        
    return max_list, mean_max, thresh, crop_t, trimmed_t
    

#%% ================== trimming both channels
def wave_harsh_peaks_all(data, ax  = None, t_base = 3000):
    
    T = len(data)
#    list_peaks = np.zeros(T)
#    list_peaks = []
    
    
    
    max_list = [data[i*t_base:(i+1)*t_base].max(axis = 0) for i in range(0,np.floor(T/t_base).astype(int))]
    mean_max = np.mean(max_list, axis =0)    
    list_peaks = np.where(np.sum(data > mean_max, axis = 1))[0]
    
#    for i_dim in range(data.shape[1]):
#        data_ch = data[:,i_dim]    
#        
#    #    for i in range(0,np.floor(T/t_base).astype(int)):
#    #        np.mean(data[i*t_base:(i+1)*t_base])
#        max_list = [max(data_ch[i*t_base:(i+1)*t_base]) for i in range(0,np.floor(T/t_base).astype(int))]
#        mean_max = np.mean(max_list)
##        thresh = mean_max*th_ratio
#    
#        list_p = np.where(data_ch>mean_max)[0]
#        list_peaks = np.append( list_peaks,list_p)
#    del list_p
    
#    plt.figure()
#    for i_f in range(2):
#        plt.subplot(2,1,i_f+1)
#        plt.plot(data[:,i_f], color = 'b')
#        plt.scatter(list_peaks, data[list_peaks.astype(int),i_f], color = 'g')    
##        plt.scatter(crop_t, data[crop_t.astype(int),i_f], color = 'g')            
    
    list_p1 = np.roll(list_peaks,1)
    list_p1[0] = 0
    del_p = (list_peaks-list_p1)
    list_p2 = [list_peaks[i] for i in np.where(del_p>1)[0]]

        
#    crop_t = []
    
    crop_t = np.unique([i for t in list_p2 for i in range(t-400,t+400)])
    
#    for t in list_p2:
#        crop_t = np.append(crop_t,np.arange(t-400,t))
#        crop_t = np.append(crop_t,np.arange(t,t+400))
#    
    crop_t = np.delete(crop_t,np.where((crop_t < 0) | (crop_t > T)))
#    crop_t = np.random.randint(1,1000,1000)
    
    trimmed_t = [i for i in range(T) if i not in crop_t]
#    plt.plot(trimmed_t, data[trimmed_t], color = 'y')
    
#    if ax is not 'silent':
#        if not ax:
#           plt.figure()
#           ax = plt
#        
#        ax.grid()
#        ax.scatter(range(len(max_list)), max_list)
#        ax.scatter(range(len(max_list)), mean_max*np.ones(len(max_list)), color = 'g')
#        ax.scatter(range(len(max_list)), thresh*np.ones(len(max_list)), color = 'r')
        
    return mean_max, trimmed_t
    
                