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
#from torch.utils.data import TensorDataset, DataLoader


import os
os.chdir('/home/bhossein/BMBF project/code_repo')

from my_data_classes import create_datasets, create_loaders
from my_net_classes import create_datasets, create_loaders


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

#%%==================== test and train splits
"creating dataset"     
test_size = 0.25

ecg_datasets = create_datasets(IDs, target, test_size, seed=seed)

print(dedent('''
             Dataset shapes:
             inputs: {}
             target: {}'''.format((ecg_datasets[0][0][0].shape[0],len(IDs)),target.shape)))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print ('device is:',device)
  
#%% ==================   Initialization              
trn_dl, val_dl, tst_dl = create_loaders(ecg_datasets, bs=128)

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

i = 0;
for i, batch in enumerate(trn_dl):
#for local_batch, local_labels in trn_dl:
    i +=1
    print(i)
print(i)