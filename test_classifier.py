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
from my_net_classes import Classifier


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
target[0:8000]=0

#%%==================== test and train splits
"creating dataset"     
test_size = 0.25

ecg_datasets = create_datasets(IDs, target, test_size, seed=seed)

print(dedent('''
             Dataset shapes:
             inputs: {}
             target: {}'''.format((ecg_datasets[0][0][0].shape[0],len(IDs)),target.shape)))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

print ('device is:',device)
  
#%% ==================   Initialization              
trn_dl, val_dl, tst_dl = create_loaders(ecg_datasets, bs=128)

raw_feat = ecg_datasets[0][0][0].shape[0]

lr = 0.001
#n_epochs = 3000
n_epochs = 300
iterations_per_epoch = len(trn_dl)
num_classes = 2
best_acc = 0
patience, trials = 500, 0
base = 1
step = 2
loss_history = []
acc_history = []

trn_sz = len(trn_dl.dataset.labels)

model = Classifier(raw_feat, num_classes).to(device)

criterion = nn.CrossEntropyLoss (reduction = 'sum')

opt = optim.Adam(model.parameters(), lr=lr)

print('Start model training')
epoch = 0

#%%===============  Learning loop
millis = round(time.monotonic() * 1000)


millis = round(time.monotonic() * 1000)
while epoch < n_epochs:
    
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(trn_dl):
        x_raw, y_batch = [t.to(device) for t in batch]
        opt.zero_grad()
        out = model (x_raw)
        loss = criterion(out,y_batch)
        epoch_loss += loss.item()
        loss.backward()
        opt.step()
    
    epoch_loss /= trn_sz
    loss_history.append(epoch_loss)


    model.eval()
    correct, total = 0, 0
    
    for batch in val_dl:
        x_raw, y_batch = [t.to(device) for t in batch]
        out = model(x_raw)
        preds = F.log_softmax(out, dim = 1).argmax(dim=1)
        total += y_batch.size(0)
        correct += (preds ==y_batch).sum().item()
        
    acc = correct / total * 100
    acc_history.append(acc)

    millis2 = round(time.monotonic() * 1000)    

    if epoch % base ==0:
#       print('Epoch: {epoch:3d}. Loss: {epoch_loss:.4f}. Acc.: {acc:2.2%}')
       print("Epoch: %3d. Loss: %4f. Acc.: %2.2f epoch-time: %d" % (epoch,epoch_loss,acc,(millis2-millis)))
       base *= step 
       
    if acc > best_acc:
        trials = 0
        best_acc = acc
        torch.save(model.state_dict(), 'best.pth')
        print('Epoch %d best model saved with accuracy: %2.2f' % (epoch,best_acc))
    else:
        trials += 1
        if trials >= patience:
            print('Early stopping on epoch %d' % (epoch))
            break
    epoch += 1

print('Done!')    

