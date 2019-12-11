#import time
#from collections import defaultdict
#from functools import partial
#from multiprocessing import cpu_count
#from pathlib import Path
#from textwrap import dedent
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd

#from sklearn.externals import joblib
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch

#from torch import nn
#from torch import optim
from torch.nn import functional as F
#from torch.optim.lr_scheduler import _LRScheduler
#from torch.utils.data import TensorDataset, DataLoader
#import datetime
#import pickle
#from git import Repo

#import os
#abspath = os.path.abspath('test_classifier_GPU_load.py')
#dname = os.path.dirname(abspath)
#os.chdir(dname)

os.chdir('/home/bhossein/BMBF project/code_repo')
#os.chdir('C:\Hinkelstien\code_repo')

#%%===============  loading a learned model
import my_net_classes
import torch
import pickle

#save_name = "1d_6con_b512_trim_2K_win_s3"
#save_name = "1d_6con_b512_trim_2K_win_s11"
#save_name = "1d_6con_b512_trim_2K_win_s1"
save_name = "1d_6con_b512_trim_2K_seed2"
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

#save_name2 = input("Input model to load (currently "+save_name+" is selected) :")
#if save_name2 != '':
#    save_name = save_name2

print(save_name + " is loaded.")

load_ECG =  torch.load ('raw_x_8K_sync_win2K.pt')
#load_ECG =  torch.load ('raw_x_8K_sync.pt') 
#load_ECG =  torch.load ('raw_x_4k_5K.pt') 
#load_ECG =  torch.load ('raw_x_all.pt') 

loaded_vars = pickle.load(open("train_"+save_name+"_variables.p","rb"))
#loaded_file = pickle.load(open("variables"+t_stamp+".p","rb"))
#loaded_file = pickle.load(open("variables_ended"+t_stamp+".p","rb"))

#cuda_num = input("cuda number:")
cuda_num = 0
device = torch.device('cuda:'+str(cuda_num) if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

raw_x = load_ECG['raw_x']
#raw_x = load_ECG['raw_x'].to(device)
target = load_ECG['target']
#target = torch.tensor(load_ECG['target']).to(device)
params = loaded_vars['params']
seed = params.seed
test_size = params.test_size
np.random.seed(seed)
t_range = params.t_range
ecg_datasets = create_datasets_file(raw_x, target, test_size, seed=seed, t_range = t_range, device = device)


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


model = my_net_classes.Classifier_1d_6_conv(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
#model = my_net_classes.Classifier_1dconv(raw_feat, num_classes, raw_size).to(device)
#model = my_net_classes.Classifier_1dconv(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
#model = my_net_classes.Classifier_1dconv_BN(raw_feat, num_classes, raw_size, batch_norm = True).to(device)

if torch.cuda.is_available()*0:
#    model.load_state_dict(torch.load("train_"+save_name+'_best.pth'))
    model.load_state_dict(torch.load("train_"+save_name+'_best.pth', map_location=lambda storage, loc: storage.cuda('cuda:'+str(cuda_num))))
else:
    model.load_state_dict(torch.load("train_"+save_name+'_best.pth', map_location=lambda storage, loc: storage))


#model = Classifier_1dconv(raw_feat, num_classes, raw_size/(2*4**3)).to(device)
#model.load_state_dict(torch.load("train_"+save_name+'_best.pth'))

model.eval()

correct, total = 0, 0

batch = []
for batch in tst_dl:
    x_raw, y_batch = batch
#    x_raw, y_batch = [t.to(device) for t in batch]
    #x_raw, y_batch = tst_ds.tensors
    #x_raw, y_batch = [t.to(device) for t in val_ds.tensors]
    out = model(x_raw)
    preds = F.log_softmax(out, dim = 1).argmax(dim=1)
    total += y_batch.size(0)
    correct += (preds ==y_batch).sum().item()
    acc = correct / total * 100

TP = ((preds ==y_batch) & (1 ==y_batch)).sum().item()
TP_rate = TP / (1 ==y_batch).sum().item() *100

FP = ((preds !=y_batch) & (0 ==y_batch)).sum().item()
FP_rate = FP / (0 ==y_batch).sum().item() *100

print('Accuracy on test data:  %2.2f' %(acc))
print('True positives on test data:  %2.2f' %(TP_rate))
print('False positives on test data:  %2.2f' %(FP_rate))

#-----------------------  visualize training curve
#f, ax = plt.subplots(1,2, figsize=(12,4))    
#ax[0].plot(loss_history, label = 'loss')
#ax[0].set_title('Validation Loss History: '+save_name)
#ax[0].set_xlabel('Epoch no.')
#ax[0].set_ylabel('Loss')
#
#ax[1].plot(smooth(acc_history, 5)[:-2], label='acc')
##ax[1].plot(acc_history, label='acc')
#ax[1].set_title('Validation Accuracy History: '+save_name)
#ax[1].set_xlabel('Epoch no.')
#ax[1].set_ylabel('Accuracy');



#checkpoint = torch.load('best_ended_11_27_17_13.pth')
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#epoch = checkpoint['epoch']
#loss = checkpoint['loss']
