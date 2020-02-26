#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison to Kai resutls

Created on Thu Feb 13 14:45:52 2020

@author: bhossein
"""

import time
#from collections import defaultdict
#from functools import partial
#from multiprocessing import cpu_count
#from pathlib import Path
#from textwrap import dedent

import pandas as pd

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

os.chdir('/home/bhossein/BMBF project/code_repo')
#os.chdir('C:\Hinkelstien\code_repo')

from my_data_classes import create_datasets_file, create_loaders, smooth, create_datasets_win
import my_net_classes
from my_net_classes import SepConv1d, _SepConv1d, Flatten, parameters
import torch
import pickle

from evaluation import evaluate
import option_utils

result_dir = 'results/' 
data_dir = 'data/' 
rep_dir = '/home/bhossein/BMBF project/Reports/'
#data_dir_hink = '/vol/hinkelstn/codes/'

import copy

# %%  read excell
kai_dir = 'C:\Hinkelstien\Reports/'
file_name = 'kai_result_10jan.xlsx'
res_db = pd.read_excel(rep_dir+file_name, index_col=0)
#pd.read_excel(open(res_dir+file_name, 'rb'))
IDs_kai = res_db['File'].values
data_tags_kai = list(res_db.index)
target_kai = res_db['Ground Truth'].values
pred_kai = res_db['Decision'].values

#%% my model abd batches

cuda_num = 0   # export CUDA_VISIBLE_DEVICES=x
device = torch.device('cuda:'+str(cuda_num) if torch.cuda.is_available() and cuda_num != 'cpu' else 'cpu')
torch.cuda.device(cuda_num)

save_name ="2d_6CN_3FC_no_BN_in_FC_long"
dataset0 = 'raw_x_8K_sync.pt'
dataset = 'raw_x_8K_sync_win2K.pt'
print("{:>40}  {:<8s}".format("Loading dataset:", dataset))

load_ECG0 = torch.load(data_dir+dataset0)
IDs = load_ECG0['IDs']

load_ECG = torch.load(data_dir+dataset)
#%%
print("{:>40}  {:<8s}".format("Loading model:", save_name))

loaded_vars = pickle.load(open(result_dir+"train_"+save_name+"_variables.p","rb"))

params = loaded_vars['params']
epoch = params.epoch
print("{:>40}  {:<8d}".format("Epoch:", epoch))
seed = params.seed
print("{:>40}  {:<8d}".format("Seed:", seed))
test_size = params.test_size
np.random.seed(seed)
t_range = params.t_range

raw_x = load_ECG['raw_x']
target = load_ECG['target']
data_tag = load_ECG['data_tag']
#data_tag = load_ECG['IDs']
if type(target) != 'torch.Tensor':
    target = torch.tensor(load_ECG['target']).to(device)

dataset_splits = create_datasets_win(raw_x, target, data_tag, test_size, seed=seed, t_range = t_range, device = device)
ecg_datasets = dataset_splits[0:3]
trn_idx, val_idx, tst_idx = dataset_splits[3:6]

trn_ds, val_ds, tst_ds = ecg_datasets

batch_size = loaded_vars['params'].batch_size
trn_dl, val_dl, tst_dl = create_loaders(ecg_datasets, bs=batch_size, jobs = 0)
#raw_feat = ecg_datasets[0][0][0].shape[0]
#raw_size = ecg_datasets[0][0][0].shape[1]
#num_classes = 2

model_path = result_dir+'train_'+save_name+'_best.pth'

#model = model_cls(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
#model0 = copy.deepcopy(model)
#
#try:
#    if torch.cuda.is_available():
#        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage.cuda(device)))
#    else:
#        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
#except:    
thresh_AF = 7
model = pickle.load(open(model_path, 'rb'))
TP_ECG, FP_ECG, list_pred_win, elapsed = evaluate(model, tst_dl, tst_idx, data_tag, thresh_AF = thresh_AF, device = device)


# %%  check kai accuracy
#TP_ECG, FP_ECG , total_P, total_N = np.zeros(4)

idx_TP_kai = ([i for i in range(len(target_kai)) if (pred_kai[i] == 1 and target_kai[i]==1)])
TP_ECG_kai = len(idx_TP_kai)
total_P = (1 == target_kai).sum()

idx_FP_kai = ([i for i in range(len(target_kai)) if (pred_kai[i] == 1 and target_kai[i]==0)])
FP_ECG_kai = len(idx_FP_kai)
total_N = (0 == target_kai).sum()

TP_ECG_rate_kai = TP_ECG_kai / total_P *100
FP_ECG_rate_kai = FP_ECG_kai / total_N *100
print("{:>40}  ".format(20*"="))
print("{:>40}  {:<8.2f}".format("Kai TP rate:", TP_ECG_rate_kai))
print("{:>40}  {:<8.2f}".format("Kai FP rate:", FP_ECG_rate_kai))

#% shared between kai and mine splits
#IDs
#IDs_kai
IDs_tst_DN = np.unique([IDs[int(data_tag[i])] for i in tst_idx])
#tst_idx_kai = [np.where(IDs_kai == j)[0].item() for j in IDs_tsts]
#tst_idx_kai = [for j in tst_idx if IDs[int(data_tag[i])]

#thresh_AF = 7
# --- Ktp_Dtp  : DN TP from kai TP 
idx_TP_DN = TP_ECG[1]
idx_FP_DN = FP_ECG[1]
IDs_TP_DN = [IDs[int(i)] for i in idx_TP_DN]
IDs_FP_DN = [IDs[int(i)] for i in idx_FP_DN]

idx_FN_kai = ([i for i in range(len(target_kai)) if (pred_kai[i] == 0 and target_kai[i]==1)])

IDs_TP_kai = [IDs_kai[i] for i in idx_TP_kai]
IDs_FP_kai = [IDs_kai[i] for i in idx_FP_kai]
IDs_FN_kai = [IDs_kai[i] for i in idx_FN_kai]

IDs_TP_tst_kai = [i for i in IDs_TP_kai if i in IDs_tst_DN]
IDs_FP_tst_kai = [i for i in IDs_FP_kai if i in IDs_tst_DN]
IDs_FN_tst_kai = [i for i in IDs_FN_kai if i in IDs_tst_DN]

IDs_Ktp_Dtp = [ID for ID in IDs_TP_tst_kai if IDs.index(ID) in idx_TP_DN]
Ktp_Dtp_rate = len(IDs_Ktp_Dtp)/len(IDs_TP_tst_kai) *100

IDs_Kfp_Dfp = [ID for ID in IDs_FP_tst_kai if IDs.index(ID) in idx_FP_DN]
Kfp_Dfp_rate = len(IDs_Kfp_Dfp)/len(IDs_FP_tst_kai) *100

IDs_Kfn_Dtp = [ID for ID in IDs_FN_tst_kai if IDs.index(ID) in idx_TP_DN]
Kfn_Dtp_rate = len(IDs_Kfn_Dtp)/len(IDs_FN_tst_kai) *100


total_p_tst = len([i for i in IDs_tst_DN if IDs.index(i) >= 8000])
total_n_tst = len([i for i in IDs_tst_DN if IDs.index(i) < 8000])

Kfn_Dtp_rate_new = (len(IDs_TP_tst_kai)+len(IDs_Kfn_Dtp))/total_p_tst *100

TP_tst_rate_kai = len(IDs_TP_tst_kai)/total_p_tst *100
FP_tst_rate_kai = len(IDs_FP_tst_kai)/total_n_tst *100

IDs_Dfn_Ktp= [i for i in IDs_TP_tst_kai if i not in IDs_TP_DN]
Dfn_Ktp_rate = len(IDs_Dfn_Ktp)/(total_p_tst-len(IDs_TP_DN))*100

IDs_TN_tst_Kai = [i for i in IDs_tst_DN if i not in (IDs_TP_tst_kai+IDs_FP_tst_kai+IDs_FN_tst_kai)]

IDs_Dfp_Ktn= [i for i in IDs_FP_DN if i in IDs_TN_tst_Kai]

Dfp_Ktn_rate = len(IDs_Dfp_Ktn)/len(IDs_FP_DN)*100

print("{:>40}  ".format(20*"="))
print("{:>40}  {:<8.2f}%".format("DN TP rate on test data:", TP_ECG[0]))
print("{:>40}  {:<8.2f}%".format("DN FP rate on test data:", FP_ECG[0]))

print("{:>40}  ".format(20*"="))
print("{:>40}  {:<8.2f}%".format("Kai TP rate on test data:", TP_tst_rate_kai))
print("{:>40}  {:<8.2f}%".format("Kai FP rate on test data:", FP_tst_rate_kai))

print("{:>40}  ".format(20*"="))
print("{:>40}  {:<8.2f}%".format("Ktp_Dtp rate:", Ktp_Dtp_rate))
print("{:>40}  {:<8.2f}%".format("Kfn_Dtp rate:", Kfn_Dtp_rate))
print("{:>40}  {:<8.2f}%".format("new TP rate on test data:", Kfn_Dtp_rate_new))
print("{:>40}  {:<8.2f}%".format("Dfn_Ktp rate:", Dfn_Ktp_rate))

print("{:>40}  ".format(20*"="))
print("{:>40}  {:<8.2f}%".format("Kfp_Dfp rate:", Kfp_Dfp_rate))
print("{:>40}  {:<8.2f}%".format("New FP rate on test data:", Kfp_Dfp_rate*FP_ECG_rate_kai/100))
print("{:>40}  {:<8.2f}%".format("Dfp_Ktn rate:", Dfp_Ktn_rate))

assert 1==2
#%%----------------- inspection
ID = IDs_Dfn_Ktp[0]
idx = IDs.index(ID)
idx_tsts = [int(i) for i in list_ECG].index(idx)
list_pred_win[idx_tsts]
idx_x = np.where(data_tag == idx)[0]

#fig, axes = plt.subplots(2, 1, sharex=True)
#for i_ax in range(2)
#    channel = raw_x[i_data,i_ch,:]
#    axes[i_ax].plot(channel,color = plt_color)


sub_dim = [3*2,5]

plt.figure(figsize=(18,10))
plt.subplots_adjust(wspace = 0.2, hspace = .5)

ch_color = ('g','r')

for i_data in range(15):
#    i_plt_ch1 = i_base+2*i_data+1
#    i_plt_ch1x = i_base+2*i_data+2
#    i_plt_ch2 = i_base+n+2*i_data+1
#    i_plt_ch2x = i_base+n+2*i_data+2
        
    
    for i_ch in range(2):
        i_plt_ch = (i_data+1)+5*i_ch+np.floor(i_data/5)*5
#        print(i_plt_ch)
        channel = raw_x[i_data,i_ch,:]
        plt.subplot(sub_dim[0],sub_dim[1],i_plt_ch)
        #plt.figure(figsize=(8,6))
        plt.plot(channel,color = ch_color[i_ch])
        plt.title('windows:'+str(i_data)+', ch:'+str(i_ch))
        plt.grid()
