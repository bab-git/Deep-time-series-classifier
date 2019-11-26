import time
from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from textwrap import dedent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader

from my_classes import Dataset

import os
os.chdir('/home/bhossein/BMBF project/code_repo')


#%% =======================
seed = 1
np.random.seed(seed)

#==================== data IDs

IDs = []
main_path = '/vol/hinkelstn/data/FILTERED/atrial_fibrillation_8k/'    
IDs.extend(os.listdir(main_path))
IDs = os.listdir(main_path)
main_path = '/vol/hinkelstn/data/FILTERED/sinus_rhythm_8k/'
IDs.extend(os.listdir(main_path))

target = np.ones(16000)
target[8000:]=2

#==================== test and train splits
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(IDs, target, test_size=0.25, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

#raw_arr = np.load('feat.npy').transpose(0, 2, 1)
#fft_arr = np.load('feat_fft.npy').transpose(0, 2, 1)
#target = np.load('target.npy')
#print(dedent('''
#             Dataset shapes:
#             raw: {} 
#             fft: {}
#             target: {}'''.format(raw_arr.shape,fft_arr.shape,target.shape)))

#%% ================== PyTorch Datasets and Data Loaders
def create_datasets(IDs, target, test_size, valid_pct=0.1, seed=None):
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
    
    trn_ds = Dataset(trn_idx,target[trn_idx])
    tst_ds = Dataset(tst_idx,target[tst_idx])
    val_ds = Dataset(val_idx,target[val_idx])
    
    return trn_ds, val_ds, tst_ds

#%% ================== 
def create_loaders(data, bs=128, jobs=0):
    """Wraps the datasets returned by create_datasets function with data loaders."""
    
    trn_ds, val_ds, tst_ds = data
    trn_dl = DataLoader(trn_ds, batch_size=bs, shuffle=True, num_workers=jobs)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=jobs)
    tst_dl = DataLoader(tst_ds, batch_size=bs, shuffle=False, num_workers=jobs)
    return trn_dl, val_dl, tst_dl         

#%% ==================    
"creating dataset"     
#trn_sz = 3810  # only the first `trn_sz` rows in each array include labelled data

ecg_data = Dataset(IDs,target)
trn_dl = DataLoader(ecg_data, batch_size=2, shuffle=0, num_workers=0)


test_size = 0.25

datasets = create_datasets(IDs, target, test_size, seed=seed)
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
  
#%% ==================                 
#raw_feat = raw_arr.shape[1]
#fft_feat = fft_arr.shape[1]

trn_dl, val_dl, tst_dl = create_loaders(datasets, bs=128)

lr = 0.001
#n_epochs = 3000
n_epochs = 3000
iterations_per_epoch = len(trn_dl)
num_classes = 9
best_acc = 0
patience, trials = 500, 0
base = 1
step = 2
loss_history = []
acc_history = []


#%%===============  Learning loop
millis = round(time.monotonic() * 1000)

for local_batch, local_labels in trn_dl:
    break