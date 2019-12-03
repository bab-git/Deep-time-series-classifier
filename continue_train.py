#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:17:33 2019

@author: bhossein
"""

import time
from textwrap import dedent
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import datetime
import pickle

import os
os.chdir('/home/bhossein/BMBF project/code_repo')

from my_data_classes import create_datasets, create_loaders
from my_net_classes import SepConv1d, _SepConv1d, Flatten, parameters, Classifier_1dconv


#%% =======================
seed = 1
np.random.seed(seed)

#==================== loading model and data
#t_stamp = input('Enter the model name to continue training: ')
t_stamp = input("enter the model to load: ")
#t_stamp = "batch_512_B"
save_name = t_stamp
#save_name = "batch_32"

n_epochs = 3000

cuda_num = input("enter cuda number to use: ")

device = torch.device('cuda:'+cuda_num if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

loaded_vars = pickle.load(open("train_"+t_stamp+"_variables.p","rb"))
#loaded_file = pickle.load(open("variables"+t_stamp+".p","rb"))
#loaded_file = pickle.load(open("variables_ended"+t_stamp+".p","rb"))

loaded_split = pickle.load(open("train_"+t_stamp+"_split.p","rb"))

ecg_datasets = loaded_split['ecg_datasets']

if ecg_datasets[0].tensors[0].device != device:
    ecg_datasets = list(ecg_datasets)
    for i in range(len(ecg_datasets)):
        X = ecg_datasets[i].tensors[0].to(device)
        y = ecg_datasets[i].tensors[1].to(device)
        ecg_datasets[i] = TensorDataset(X,y)

trn_ds, val_ds, tst_ds = ecg_datasets
raw_feat = ecg_datasets[0][0][0].shape[0]
raw_size = ecg_datasets[0][0][0].shape[1]
num_classes = 2

model = Classifier_1dconv(raw_feat, num_classes, raw_size/(2*4**3)).to(device)
model.load_state_dict(torch.load("train_"+t_stamp+'_best.pth'))

#print(dedent('''
#             Dataset shapes:
#             inputs: {}
#             target: {}'''.format((ecg_datasets[0][0][0].shape,len(IDs)),target.shape)))



print ('device is:',device)

#%% ==================   Networkj parameters
batch_size = loaded_vars['params'].batch_size
#batch_size = 512

trn_dl, val_dl, tst_dl = create_loaders(ecg_datasets, bs=batch_size, jobs = 0)

t_range = loaded_vars['params'].t_range
lr = loaded_vars['params'].lr
#n_epochs = 3000

#iterations_per_epoch = len(trn_dl)
num_classes = 2
acc_history = loaded_vars['acc_history']
best_acc = max(acc_history)
patience = loaded_vars['params'].patience
trials = 0
base = 1
step = loaded_vars['params'].step
loss_history = []

trn_sz = len(ecg_datasets[0])

criterion = nn.CrossEntropyLoss (reduction = 'sum')

opt = optim.Adam(model.parameters(), lr=lr)

print('Continue training for model: '+t_stamp+" with best val-acc of %2.2f" % (best_acc))
epoch = loaded_vars['params'].epoch



#%%===============  Learning loop
#millis = round(time.time())


#millis = round(time.monotonic() * 1000)
while epoch < n_epochs:
    
    model.train()
    epoch_loss = 0
    millis = (time.time())
    
#    print('trainig....')
#    for batch in trn_dl.dataset:
#        break
    for i, batch in enumerate(trn_dl):
#        break
        x_raw, y_batch = batch
#        x_raw, y_batch = [t.to(device) for t in trn_ds.tensors]
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
    
#    print('validation....')
#    for batch in val_dl:
#        x_raw, y_batch = [t.to(device) for t in batch]
    x_raw, y_batch = [t.to(device) for t in val_ds.tensors]
    out = model(x_raw)
    preds = F.log_softmax(out, dim = 1).argmax(dim=1)
    total += y_batch.size(0)
    correct += (preds ==y_batch).sum().item()
        
    acc = correct / total * 100
    acc_history.append(acc)

    millis2 = (time.time())

    if epoch % base ==0:
#       print('Epoch: {epoch:3d}. Loss: {epoch_loss:.4f}. Acc.: {acc:2.2%}')
       print("model: "+save_name+" - Epoch %3d. Loss: %4f. Acc.: %2.2f epoch-time: %4.2f" % (epoch,epoch_loss,acc,(millis2-millis)))
       base *= step 
       
    if acc > best_acc:
        print("model: "+save_name+" - Epoch %d best model being saved with accuracy: %2.2f" % (epoch,best_acc))
        trials = 0
        best_acc = acc
#        torch.save(model.state_dict(), 'best.pth')        
        torch.save(model.state_dict(), "train_"+save_name+'_best.pth')
#        pickle.dump({'epoch':epoch,'acc_history':acc_history},open("train_"+save_name+"variables.p","wb"))
        params = parameters(lr, epoch, patience, step, batch_size, t_range)
        pickle.dump({'params':params,'acc_history':acc_history, 'loss_history':loss_history},open("train_"+save_name+"_variables.p","wb"))
    else:
        trials += 1
        if trials >= patience:
            print('Early stopping on epoch %d' % (epoch))
            break
    epoch += 1

#now = datetime.datetime.now()
#date_stamp = str(now.strftime("_%m_%d_%H_%M"))
#torch.save(model.state_dict(), 'best_ended_'+save_name+date_stamp+'.pth')
#params = parameters(lr, epoch, patience, step, batch_size, t_range)
#pickle.dump({'ecg_datasets':ecg_datasets},open("train_"+save_name+"_split.p","wb"))

print("Model is saved to: "+"train_"+save_name+'_best.pth')

#-----git push
#if os.path.isfile(load_file):
#repo = Repo(os.getcwd())
#repo.index.add(["variables_ended.p"])
#repo.index.add(["best_ended.pth"])
##repo.index.add(["variables.p"])
#repo.index.commit("Training finished on: "+str(datetime.datetime.now()))
#origin = repo.remotes.origin
#origin.push()

print('Done!')    

#%%==========================  test result
test_results = []
model.load_state_dict(torch.load('best_best.pth'))
model.eval()

correct, total = 0, 0

batch = []
for batch in tst_dl:
        x_raw, y_batch = [t.to(device) for t in batch]
        out = model(x_raw)
        preds = F.log_softmax(out, dim = 1).argmax(dim=1)
        total += y_batch.size(0)
        correct += (preds ==y_batch).sum().item()
        
acc = correct / total * 100

print('Accuracy on test data: ',acc)