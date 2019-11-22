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

import os
os.chdir('/home/bhossein/BMBF project/code_repo')


#%% =======================
seed = 1
np.random.seed(seed)

raw_arr = np.load('feat.npy').transpose(0, 2, 1)
fft_arr = np.load('feat_fft.npy').transpose(0, 2, 1)
target = np.load('target.npy')
print(dedent('''
             Dataset shapes:
             raw: {} 
             fft: {}
             target: {}'''.format(raw_arr.shape,fft_arr.shape,target.shape)))

#%% ================== PyTorch Datasets and Data Loaders
def create_datasets(data, target, train_size, valid_pct=0.1, seed=None):
    """Converts NumPy arrays into PyTorch datsets.
    
    Three datasets are created in total:
        * training dataset
        * validation dataset
        * testing (un-labelled) dataset
    """
    raw, fft = data
    assert len(raw) == len(fft)
    sz = train_size
    idx = np.arange(sz)
    trn_idx, val_idx = train_test_split(
        idx, test_size=valid_pct, random_state=seed)
    
    trn_ds = TensorDataset(
        torch.tensor(raw[:sz][trn_idx]).float(), 
        torch.tensor(fft[:sz][trn_idx]).float(), 
        torch.tensor(target[:sz][trn_idx]).long())
    val_ds = TensorDataset(
        torch.tensor(raw[:sz][val_idx]).float(), 
        torch.tensor(fft[:sz][val_idx]).float(), 
        torch.tensor(target[:sz][val_idx]).long())
    tst_ds = TensorDataset(
        torch.tensor(raw[sz:]).float(), 
        torch.tensor(fft[sz:]).float(), 
        torch.tensor(target[sz:]).long())
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
class _SepConv1d(nn.Module):
    """  simple separable convolution implementation.
    
    The separable convlution is a method to reduce number of the parameters 
    in the deep learning network for slight decrease in predictions quality.
    """
    
    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding = pad, groups = ni)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1)
        
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

#%% ==================         
class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    
    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride, pad, drop=None,
                 activ=lambda: nn.ReLU(inplace=True)):
    
        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.layers(x)
        
#%% ==================         
class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)
#%% ==================         
class Classifier(nn.Module):
    def __init__(self, raw_ni, fft_ni, no, drop=.5):
        super().__init__()
        
        self.raw = nn.Sequential(
            SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop),
            SepConv1d(    32,  64, 8, 4, 2, drop=drop),
            SepConv1d(    64, 128, 8, 4, 2, drop=drop),
            SepConv1d(   128, 256, 8, 4, 2),
            Flatten(),
            nn.Dropout(drop), nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))
        
        self.fft = nn.Sequential(
            SepConv1d(fft_ni,  32, 8, 2, 4, drop=drop),
            SepConv1d(    32,  64, 8, 2, 4, drop=drop),
            SepConv1d(    64, 128, 8, 4, 4, drop=drop),
            SepConv1d(   128, 128, 8, 4, 4, drop=drop),
            SepConv1d(   128, 256, 8, 2, 3),
            Flatten(),
            nn.Dropout(drop), nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, no))
        
    def forward(self, t_raw, t_fft):
        raw_out = self.raw(t_raw)
        fft_out = self.fft(t_fft)
        t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(t_in)
        return out
    
#%% ==================         
trn_sz = 3810  # only the first `trn_sz` rows in each array include labelled data

datasets = create_datasets((raw_arr, fft_arr), target, trn_sz, seed=seed)
  
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

#%% ==================                 
raw_feat = raw_arr.shape[1]
fft_feat = fft_arr.shape[1]

trn_dl, val_dl, tst_dl = create_loaders(datasets, bs=256)

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

model = Classifier(raw_feat, fft_feat, num_classes).to(device)

criterion = nn.CrossEntropyLoss (reduction = 'sum')

opt = optim.Adam(model.parameters(), lr=lr)

print('Start model training')
epoch = 0

#%% 
millis = round(time.monotonic() * 1000)
while epoch < n_epochs:
    
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(trn_dl):
        x_raw, x_fft, y_batch = [t.to(device) for t in batch]
        opt.zero_grad()
        out = model (x_raw,x_fft)
        loss = criterion(out,y_batch)
        epoch_loss += loss.item()
        loss.backward()
        opt.step()
    
    epoch_loss /= trn_sz
    loss_history.append(epoch_loss)
    
    model.eval()
    correct, total = 0, 0
    
    for batch in val_dl:
        x_raw, x_fft, y_batch = [t.to(device) for t in batch]
        out = model(x_raw, x_fft)
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

#%%===========================        
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode = 'same')
    return y_smooth

#%%===========================        
f, ax = plt.subplots(1,2, figsize=(12,4))    
ax[0].plot(loss_history, label = 'loss')
ax[0].set_title('Validation Loss History')
ax[0].set_xlabel('Epoch no.')
ax[0].set_ylabel('Loss')

ax[1].plot(smooth(acc_history, 5)[:-2], label='acc')
#ax[1].plot(acc_history, label='acc')
ax[1].set_title('Validation Accuracy History')
ax[1].set_xlabel('Epoch no.')
ax[1].set_ylabel('Accuracy');

#%%===========================        
test_results = []
model.load_state_dict(torch.load('best.pth'))
model.eval()

#for batches in val_dl:
x_raw=[]
for x_raw, x_fft, _ in val_dl:
    1
    #for x_raw, x_fft, _ in enumtst_dl:
#    x_raw, x_fft, _ = [t.to(device) for t in batches]
    batches = [t.to(device) for t in (x_raw, x_fft)]
    out = model(*batches)
    y_hat = F.log_softmax(out, dim=1).argmax(dim=1)
    test_results.extend(y_hat.tolist())