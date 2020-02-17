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
save_name           = option_utils.find_save(model_name, data_name)
if save_name == 'NONE':
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

#device = ecg_datasets[0].tensors[0].device
#device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')