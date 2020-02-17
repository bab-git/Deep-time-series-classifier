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
from torch.utils.data import TensorDataset, DataLoader

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
data_dir_hink = '/vol/hinkelstn/codes/'

import copy

#%% ============== options
model_cls, model_name   = option_utils.show_model_chooser()
dataset, data_name  = option_utils.show_data_chooser()
#save_name           = option_utils.find_save(model_name, data_name)
#save_name ="2d_6CN_3FC_no_BN_in_FC_long"
#if save_name == 'NONE':
save_name ="2d_6CN_3FC_no_BN_in_FC_long"
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


# %% ================ loading data


#dataset0 = 'raw_x_8K_sync.pt'
dataset0 = 'raw_x_8K_sync_win2K.pt'

print("{:>40}  {:<8s}".format("Loading dataset:", dataset0))

load_ECG0 = torch.load(data_dir_hink+dataset0)
IDs0 = load_ECG0['IDs']
data_tag0 = load_ECG0['data_tag']

#dataset1 = 'raw_x_8K_sync_win2K.pt'
#load_ECG1 = torch.load(data_dir_hink+dataset1)


print("{:>40}  {:<8s}".format("Loading dataset:", dataset))

load_ECG = torch.load(data_dir_hink+dataset)

#%%===============  loading experiment's parameters and batches

print("{:>40}  {:<8s}".format("Loading model:", model_name))




loaded_vars = pickle.load(open(result_dir+"train_"+save_name+"_variables.p","rb"))
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

#cuda_num = input("cuda number:")
#cuda_num = 0   # export CUDA_VISIBLE_DEVICES=x

#device = torch.device('cuda:'+str(cuda_num) if torch.cuda.is_available() and cuda_num != 'cpu' else 'cpu')
#device = torch.device('cpu')

raw_x = load_ECG['raw_x']
#raw_x = load_ECG['raw_x'].to(device)
target = load_ECG['target']
data_tag = load_ECG['data_tag']
IDs = load_ECG['IDs'] if 'IDs' in load_ECG else []

if type(target) != 'torch.Tensor':
    target = torch.tensor(load_ECG['target']).to(device)


# %%
raw_x0 = load_ECG0['raw_x']    
target0 = torch.tensor(load_ECG0['target'])
#if type(target0) != 'torch.Tensor':
#    target0 = 
    
dataset_splits = create_datasets_win(raw_x0, target0, data_tag0, test_size, seed=seed, t_range = t_range)
ecg_datasets = dataset_splits[0:3]
trn_idx, val_idx, tst_idx = dataset_splits[3:6]
trn_ds, val_ds, tst_ds = ecg_datasets

batch_size = loaded_vars['params'].batch_size
trn_dl, val_dl, tst_dl = create_loaders(ecg_datasets, bs=batch_size, jobs = 0)
raw_feat = ecg_datasets[0][0][0].shape[0]
raw_size = ecg_datasets[0][0][0].shape[1]
num_classes = 2


IDs_tst_DN = np.unique([IDs0[int(data_tag0[i])] for i in tst_idx])

extend_idx = lambda IDs,data_tag,IDs_tst_DN: [j for i in IDs_tst_DN for j in np.where(data_tag == IDs.index(i))[0]]
   
tst_idx_new = extend_idx(IDs, data_tag, IDs_tst_DN)
tst_ds_new = TensorDataset(raw_x[tst_idx_new,:,t_range.start:t_range.stop].float().to(device),
                           target[tst_idx_new].long().to(device))
tst_dl_new = DataLoader(tst_ds_new, batch_size=batch_size, shuffle=False)



path_data_n = '/vol/hinkelstn/data/FILTERED/sinus_rhythm_8k/'    
path_data_AF = '/vol/hinkelstn/data/FILTERED/atrial_fibrillation_8k/'
IDs_n = os.listdir(path_data_n)
IDs_AF = os.listdir(path_data_AF)
target_ECG = [1 if i in IDs_AF else 0 for i in IDs]
#load_ECG0['target_ECG'] = target_ECG
#torch.save(load_ECG0, dataset0)

# %% ===============  model

model_path = result_dir+'train_'+save_name+'_best.pth'

model = model_cls(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
model0 = copy.deepcopy(model)

try:
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage.cuda(device)))
    else:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
except:    
    model = pickle.load(open(model_path, 'rb'))
    
#%% evaluation Acc
thresh_AF = 3

#TP_ECG_rate, FP_ECG_rate, list_pred_win, elapsed = evaluate(model, tst_dl, tst_idx, data_tag, thresh_AF = thresh_AF, device = device)
#acc, list_pred_win = evaluate(model, tst_dl, tst_idx, data_tag0, thresh_AF = thresh_AF, device = device, acc_eval = True)
acc, list_pred = evaluate(model, tst_dl_new, tst_idx_new, data_tag, thresh_AF = thresh_AF, device = device, acc_eval = True)

#%% evaluation ECG splits
list_wins = [len(np.where(data_tag==i_ecg)[0]) for i_ecg in range(16000)]
print(np.unique(list_wins))
print(list_wins.index(0))

tst_idx = tst_idx_new

#win_size = (data_tag==0).sum()
win_size  = 33
# thresh_AF = win_size /2
# thresh_AF = 3

list_ECG = [int(i) for i in np.unique([data_tag[i] for i in tst_idx])]
#list_ECG2 = [IDs.index(i) for i in IDs_tst_DN]
#list_ECG = np.unique([data_tag[i] for i in tst_idx if target[i] == label])
#len(list_error_ECG)/8000*100

TP_ECG, FP_ECG , total_P, total_N = np.zeros(4)
idx_TP = []
idx_FP = []
list_pred_win = 100*np.ones([len(list_ECG), win_size])
for i_row, i_ecg in enumerate(list_ECG):
    list_win = np.where(data_tag==i_ecg)[0]
    pred_win = [list_pred[tst_idx.index(i)] for i in list_win]
#    print(pred_win)
    list_pred_win[i_row,:] = pred_win
                        
    if target_ECG[i_ecg] == 1:   #AF
        total_P += 1
        if (np.array(pred_win)==1).sum() >= thresh_AF:
            TP_ECG += 1
            idx_TP = np.append(idx_TP,i_ecg)
    else:         # normal
        total_N += 1
        if (np.array(pred_win)==1).sum() >= thresh_AF:
            FP_ECG += 1
            idx_FP = np.append(idx_FP,i_ecg)
            
    
#TP_ECG_rate = TP_ECG / len(list_ECG) *100
TP_ECG_rate = TP_ECG / total_P *100
FP_ECG_rate = FP_ECG / total_N *100

flops, params = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False)

#    print("{:>40}  {:<8.2f}".format("Accuracy on all windows of test data:", acc))

print("{:>40}  {:<8d}".format("Threshold for detecting AF:", thresh_AF))
print("{:>40}  {:<8.3f}".format("TP rate:", TP_ECG_rate))
print("{:>40}  {:<8.3f}".format("FP rate:", FP_ECG_rate))

print('{:>40}  {:<8d}'.format('Number of parameters:', params))
print('{:>40}  {:<8.0f}'.format('Computational complexity:', flops))

#return (TP_ECG_rate,idx_TP), (FP_ECG_rate,idx_FP), list_pred_win, elapsed