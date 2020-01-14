import time
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
from torch import nn
from torch import optim
from torch.nn import functional as F
#from torch.optim.lr_scheduler import _LRScheduler
#from torch.utils.data import TensorDataset, DataLoader
#import datetime
import pickle
#from git import Repo

from torchsummary import summary

import os
#abspath = os.path.abspath('test_classifier_GPU_load.py')
#dname = os.path.dirname(abspath)
#os.chdir(dname)

os.chdir('/home/bhossein/BMBF project/code_repo')
#os.chdir('C:\Hinkelstien\code_repo')

from my_data_classes import create_datasets, create_loaders, read_data, create_datasets_file, create_datasets_win, smooth
import my_net_classes
from my_net_classes import SepConv1d, _SepConv1d, Flatten, parameters


#%% =======================
seed = 1
seed2 = input ('Enter seed value for randomizing the splits (default = 1):')
if seed2 != '':
    seed = int(seed2)

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

#t_range = range(1000,1512)

t_win = 2**11  #2048
#t_shift = 400
t_shift = None

t_range = range(t_win)
#load_ECG =  torch.load ('raw_x_all.pt') 
#load_ECG =  torch.load ('raw_x_40k_50K.pt') 
#load_ECG =  torch.load ('raw_x_6K.pt') 
#load_ECG =  torch.load ('raw_x_8K_sync.pt')
load_ECG =  torch.load ('raw_x_8K_sync_win2K.pt')
            

#%%==================== test and train splits
"creating dataset"     
#test_size = 0.25   
test_size = 0.3    #default

#cuda_num = input("enter cuda number to use: ")
cuda_num = 0   # export CUDA_VISIBLE_DEVICES=0

device = torch.device('cuda:'+str(cuda_num) if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

if t_shift:
    raw_x = load_ECG['raw_x']    
    length = raw_x.shape[2]
    n_win = np.floor(length / t_shift) - np.floor(t_win / t_shift)    
    raw_x_extend = torch.empty([raw_x.shape[0]*n_win.astype(int),raw_x.shape[1],t_win])
    target_extend = torch.empty(raw_x.shape[0]*n_win.astype(int))    
    target = load_ECG['target']

    print("Extracting temporal windows from ECG files..." )
                                      
    for i_data in range(raw_x.shape[0]):    
#        if i_data % 500 ==0:
#            print("data number: "+str(i_data))
        for i_win in range(int(n_win)):
                    
            i_ext = i_data*n_win+i_win
            
            raw_x_extend[int(i_ext),:,:] = raw_x[i_data,:,i_win*t_shift:i_win*t_shift+t_win] 
            target_extend[int(i_ext)] = target[i_data]                
    
    del raw_x, target
    raw_x = raw_x_extend.to(device)
    target = target_extend.to(device)
    
else:

    raw_x = load_ECG['raw_x']
#    raw_x = load_ECG['raw_x'].to(device)
    #raw_x.pin_memory = True
#    target = load_ECG['target']
    target = torch.tensor(load_ECG['target'])
    
    data_tag = load_ECG['data_tag']


dataset_splits = create_datasets_win(raw_x, target, data_tag, test_size, seed=seed, t_range = t_range, device = device)
ecg_datasets = dataset_splits[0:3]
trn_idx, val_idx, tst_idx = dataset_splits[3:6]


#ecg_datasets = create_datasets_file(raw_x, target, test_size, seed=seed, t_range = t_range, device = device)
#trn_ds, val_ds, tst_ds = create_datasets_file(raw_x, target, test_size, seed=seed, t_range = t_range)
trn_ds, val_ds, tst_ds = ecg_datasets

#ecg_datasets = create_datasets(IDs, target, test_size, seed=seed, t_range = t_range)

#print(dedent('''
#             Dataset shapes:
#             inputs: {}
#             target: {}'''.format((ecg_datasets[0][0][0].shape,len(IDs)),target.shape)))



print ('device is loaded to device:',device)
#%% ==================   Initialization              


batch_size = (2**3)*64   #default = 512
val_batch_size = batch_size  #default
#val_batch_size = 4 * batch_size 
#batch_size = 64
#batch_size = (2**5)*64
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

trn_dl, val_dl, tst_dl = create_loaders(ecg_datasets, bs=batch_size, jobs = 0, bs_val = val_batch_size)

raw_feat = trn_ds[0][0].shape[0]
raw_size = trn_ds[0][0].shape[1]
trn_sz = len(trn_ds)

model = my_net_classes.Classifier_1d_6_conv(raw_feat, num_classes, raw_size, 
                                            batch_norm = True, conv_type = '2d').to(device)
#model = my_net_classes.Classifier_1d_6_conv(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
#model = my_net_classes.Classifier_1dconv(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
#model = my_net_classes.Classifier_1dconv_BN(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
#model = Classifier_1dconv(raw_feat, num_classes, raw_size/(2*4**3)).to(device)
#model = Classifier_1dconv(raw_feat, num_classes, raw_size).to(device)

#model2 = model.to('cpu')
#summary(model2, input_size=(raw_feat, raw_size), batch_size = batch_size, device = 'cpu')

#if torch.cuda.device_count() > 1:
#    print("Let's use", torch.cuda.device_count(), "GPUs!")    
#    model = nn.DataParallel(model,device_ids=[0,1,5]).cuda()



criterion = nn.CrossEntropyLoss (reduction = 'sum')

opt = optim.Adam(model.parameters(), lr=lr)

#print('Enter a save-file name for this trainig:')
print("chosen batch size: %d, test size: %2.2f" % (batch_size, test_size))
save_name = input("Enter a save-file name for this trainig: ")

print('Start model training')
epoch = 0

#pickle.dump({'ecg_datasets':ecg_datasets},open("train_"+save_name+"_split.p","wb"))
#torch.save(ecg_datasets, 'train_'+save_name+'.pth')
#%%===============  Learning loop`
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

#    with torch.no_grad():
    model.eval()
    correct, total = 0, 0
    
#    print('validation....')
    for batch in val_dl:
        x_raw, y_batch = batch
#        x_raw, y_batch = [t.to(device) for t in batch]
#    x_raw, y_batch = [t.to(device) for t in val_ds.tensors]
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
            model.load_state_dict(torch.load("train_"+save_name+'_best.pth'))
            model.opt = opt
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

atch = []
i_error = []
list_pred = []
for i_batch, batch in enumerate(tst_dl):
    x_raw, y_batch = batch
    list_x = list(range(i_batch*tst_dl.batch_size,min((i_batch+1)*tst_dl.batch_size,len(tst_ds))))
#    x_raw, y_batch = [t.to(device) for t in batch]
#    x_raw, y_batch = tst_ds.tensors
    #x_raw, y_batch = [t.to(device) for t in val_ds.tensors]
    out = model(x_raw)
    preds = F.log_softmax(out, dim = 1).argmax(dim=1)
    list_pred = np.append(list_pred,preds)
    total += y_batch.size(0)
    correct += (preds ==y_batch).sum().item()    
#    i_error = np.append(i_error,np.where(preds !=y_batch))
    i_error = np.append(i_error,[list_x[i] for i in np.where(preds !=y_batch)[0]])    
#    TP += ((preds ==y_batch) & (1 ==y_batch)).sum().item()
#    total_P += (1 ==y_batch).sum().item()
#    FP += ((preds !=y_batch) & (0 ==y_batch)).sum().item()

acc = correct / total * 100
print('Accuracy on all windows of test data:  %2.2f' %(acc))

win_size = (data_tag==0).sum()
#thresh_AF = win_size /2
thresh_AF = 3

list_ECG = np.unique([data_tag[i] for i in tst_idx])
#list_ECG = np.unique([data_tag[i] for i in tst_idx if target[i] == label])
#len(list_error_ECG)/8000*100

TP_ECG, FN_ECG , total_P, total_N = np.zeros(4)
list_pred_win = 100*np.ones([len(list_ECG), win_size])
for i_row, i_ecg in enumerate(list_ECG):
    list_win = np.where(data_tag==i_ecg)[0]
    pred_win = [list_pred[tst_idx.index(i)] for i in list_win]
#    print(pred_win)
    list_pred_win[i_row,:] = pred_win    
                        
    if i_ecg >8000:   #AF
        total_P +=1
        if (np.array(pred_win)==1).sum() >= thresh_AF:
            TP_ECG += 1                    
    else:         # normal
        total_N +=1
        if (np.array(pred_win)==1).sum() >= thresh_AF:
            FN_ECG += 1
            
    
#TP_ECG_rate = TP_ECG / len(list_ECG) *100
TP_ECG_rate = TP_ECG / total_P *100
FN_ECG_rate = FN_ECG / total_N *100


print("Threshold for detecting AF: %d" % (thresh_AF))
print("TP rate: %2.3f" % (TP_ECG_rate))
print("FN rate: %2.3f" % (FN_ECG_rate))


assert 1==2
#%%===============  checking internal values
drop=.5
batch_norm = True
model1 = nn.Sequential(
             _SepConv1d(2,  32, 8, 2, 3 )  #out: raw_size/str
#            SepConv1d(2,  32, 8, 2, 3, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
#            SepConv1d(    32,  64, 8, 4, 2, drop=drop, batch_norm = batch_norm),
#            SepConv1d(    64, 128, 8, 4, 2, drop=drop, batch_norm = batch_norm),
#            SepConv1d(   128, 256, 8, 4, 2, drop=drop, batch_norm = batch_norm),
#            SepConv1d(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm),
#            SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
#            Flatten(),
#            nn.Linear(256, 64), nn.ReLU(inplace=True),
#            nn.Linear( 64, 64), nn.ReLU(inplace=True)
            ).to(device)
model_out = model1(x_raw)
#model_out = model1(x_raw[0,:,:])
x_raw.shape
model_out.shape


model2 = model.to('cpu')
summary(model2, input_size=(raw_feat, raw_size), batch_size = batch_size, device = 'cpu')