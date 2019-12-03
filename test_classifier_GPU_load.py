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
#from torch.optim.lr_scheduler import _LRScheduler
#from torch.utils.data import TensorDataset, DataLoader
#import datetime
import pickle
#from git import Repo

import os
os.chdir('/home/bhossein/BMBF project/code_repo')

from my_data_classes import create_datasets, create_loaders, read_data, create_datasets_file, smooth
import my_net_classes
from my_net_classes import SepConv1d, _SepConv1d, Flatten, parameters


#%% =======================
seed = 1
np.random.seed(seed)

#==================== data IDs

#IDs = []
#main_path = '/vol/hinkelstn/data/FILTERED/atrial_fibrillation_8k/'    
#IDs.extend(os.listdir(main_path))
#IDs = os.listdir(main_path)
#main_path = '/vol/hinkelstn/data/FILTERED/sinus_rhythm_8k/'
#IDs.extend(os.listdir(main_path))
#
#target = np.ones(16000)
#target[0:8000]=0

t_range = range(40000,40512)

#%%==================== test and train splits
"creating dataset"     
test_size = 0.25

cuda_num = input("enter cuda number to use: ")

device = torch.device('cuda:'+cuda_num if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

load_ECG =  torch.load ('raw_x_all.pt') 
raw_x = load_ECG['raw_x'].to(device)
#raw_x.pin_memory = True
target = torch.tensor(load_ECG['target']).to(device)


ecg_datasets = create_datasets_file(raw_x, target, test_size, seed=seed, t_range = t_range)
#trn_ds, val_ds, tst_ds = create_datasets_file(raw_x, target, test_size, seed=seed, t_range = t_range)
trn_ds, val_ds, tst_ds = ecg_datasets

#ecg_datasets = create_datasets(IDs, target, test_size, seed=seed, t_range = t_range)

#print(dedent('''
#             Dataset shapes:
#             inputs: {}
#             target: {}'''.format((ecg_datasets[0][0][0].shape,len(IDs)),target.shape)))



print ('device is:',device)


#%% ==================   Initialization              

#batch_size = 64
batch_size = (2**3)*64
#batch_size = (2**2)*64
#batch_size = "full"
lr = 0.001
#n_epochs = 3000
n_epochs = 10000
#iterations_per_epoch = len(trn_dl)
num_classes = 2
best_acc = 0
patience, trials = 500, 0
base = 1
step = 2
loss_history = []
acc_history = []

trn_dl, val_dl, tst_dl = create_loaders(ecg_datasets, bs=batch_size, jobs = 0)

raw_feat = trn_ds[0][0].shape[0]
raw_size = trn_ds[0][0].shape[1]
trn_sz = len(trn_ds)

model = my_net_classes.Classifier_1dconv(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
#model = my_net_classes.Classifier_1dconv_BN(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
#model = Classifier_1dconv(raw_feat, num_classes, raw_size/(2*4**3)).to(device)
#model = Classifier_1dconv(raw_feat, num_classes, raw_size).to(device)


#if torch.cuda.device_count() > 1:
#    print("Let's use", torch.cuda.device_count(), "GPUs!")    
#    model = nn.DataParallel(model,device_ids=[0,1,5]).cuda()



criterion = nn.CrossEntropyLoss (reduction = 'sum')

opt = optim.Adam(model.parameters(), lr=lr)

#print('Enter a save-file name for this trainig:')
print("chosen batch size: %d" % (batch_size))
save_name = input("Enter a save-file name for this trainig: ")

print('Start model training')
epoch = 0

#pickle.dump({'ecg_datasets':ecg_datasets},open("train_"+save_name+"_split.p","wb"))
torch.save(ecg_datasets, 'train_'+save_name+'.pth')
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
        params = parameters(lr, epoch, patience, step, batch_size, t_range, seed, test_size)
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


#%%===========================        
#def smooth(y, box_pts):
#    box = np.ones(box_pts)/box_pts
#    y_smooth = np.convolve(y, box, mode = 'same')
#    return y_smooth

#%%==========================  visualize training curve
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

#%%==========================  test result
test_results = []
#model.load_state_dict(torch.load('best_ended_11_29_13_35.pth'))
model.eval()

correct, total = 0, 0

batch = []
#for batch in tst_dl:
x_raw, y_batch = [t.to(device) for t in tst_ds.tensors]
out = model(x_raw)
preds = F.log_softmax(out, dim = 1).argmax(dim=1)
total += y_batch.size(0)
correct += (preds ==y_batch).sum().item()
        
acc = correct / total * 100

print('Accuracy on test data:  %2.2f' %(acc))


assert 1==2
#%%===============  Learning loop
model1 = nn.Sequential(
            SepConv1d(     2,  32, 8, 2, 3, drop=drop),
            SepConv1d(    32,  64, 8, 4, 2, drop=drop),
            SepConv1d(    64, 128, 8, 4, 2, drop=drop),
            SepConv1d(   128, 256, 8, 4, 2),
            Flatten(),
#            nn.Linear(256, 64), nn.ReLU(inplace=True),
#            nn.Linear( 64, 64), nn.ReLU(inplace=True)
            ).to(device)
model_out = model1(x_raw)
model_out.shape

#%%===============  loading a learned model
import my_net_classes

save_name = "1dconv_b512_drop1B"
#save_name = "1dconv_b512_drop1"
#save_name = "batch_512_BN_B"
#save_name = "1dconv_b512_BNM_B"
#save_name = "1dconv_b512_BNA_B"
#save_name = "batch_512_BNA"
#save_name = "batch_512_BN"
#save_name = "batch_512_B"
#t_stamp = "_batch_512_11_29_17_03"

loaded_vars = pickle.load(open("train_"+save_name+"_variables.p","rb"))
#loaded_file = pickle.load(open("variables"+t_stamp+".p","rb"))
#loaded_file = pickle.load(open("variables_ended"+t_stamp+".p","rb"))

#loaded_split = pickle.load(open("train_"+save_name+"_split.p","rb"))
#ecg_datasets = torch.load('train_'+save_name+'.pth', map_location=lambda storage, loc: storage.cuda('cuda:'+str(cuda_num)))
device = torch.device('cuda:'+str(cuda_num) if torch.cuda.is_available() else 'cpu')

load_ECG =  torch.load ('raw_x_all.pt') 
raw_x = load_ECG['raw_x'].to(device)
target = torch.tensor(load_ECG['target']).to(device)
params = loaded_vars['params']
seed = params.seed
test_size = params.test_size
np.random.seed(seed)
t_range = params.t_range
ecg_datasets = create_datasets_file(raw_x, target, test_size, seed=seed, t_range = t_range)


acc_history = loaded_vars['acc_history']
loss_history = loaded_vars['loss_history']
#ecg_datasets = loaded_split['ecg_datasets']
trn_ds, val_ds, tst_ds = ecg_datasets
batch_size = loaded_vars['params'].batch_size
trn_dl, val_dl, tst_dl = create_loaders(ecg_datasets, bs=batch_size, jobs = 0)
raw_feat = ecg_datasets[0][0][0].shape[0]
raw_size = ecg_datasets[0][0][0].shape[1]
num_classes = 2

device = ecg_datasets[0].tensors[0].device
#device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')


model = my_net_classes.Classifier_1dconv(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
#model = my_net_classes.Classifier_1dconv_BN(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
model.load_state_dict(torch.load("train_"+save_name+'_best.pth'))

#model = Classifier_1dconv(raw_feat, num_classes, raw_size/(2*4**3)).to(device)
#model.load_state_dict(torch.load("train_"+save_name+'_best.pth'))

model.eval()

correct, total = 0, 0

batch = []
x_raw, y_batch = [t.to(device) for t in tst_ds.tensors]
#x_raw, y_batch = [t.to(device) for t in val_ds.tensors]
out = model(x_raw)
preds = F.log_softmax(out, dim = 1).argmax(dim=1)
total += y_batch.size(0)
correct += (preds ==y_batch).sum().item()
        
acc = correct / total * 100
print('Accuracy on test data:  %2.2f' %(acc))

#-----------------------  visualize training curve
f, ax = plt.subplots(1,2, figsize=(12,4))    
ax[0].plot(loss_history, label = 'loss')
ax[0].set_title('Validation Loss History: '+save_name)
ax[0].set_xlabel('Epoch no.')
ax[0].set_ylabel('Loss')

ax[1].plot(smooth(acc_history, 5)[:-2], label='acc')
#ax[1].plot(acc_history, label='acc')
ax[1].set_title('Validation Accuracy History: '+save_name)
ax[1].set_xlabel('Epoch no.')
ax[1].set_ylabel('Accuracy');



#checkpoint = torch.load('best_ended_11_27_17_13.pth')
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#epoch = checkpoint['epoch']
#loss = checkpoint['loss']