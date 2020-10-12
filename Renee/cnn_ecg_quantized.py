#import argparse
#import glob
#import os
import pickle

#import my_net_classes
from my_data_classes import create_datasets_file, create_loaders, smooth,\
    create_datasets_win, create_datasets_cv
    
import numpy as np
#import pandas as pd
#import wavio
#from scipy import stats

import torch
from torch.utils.data import TensorDataset, DataLoader

from torch.utils import data
#from torch.utils.data import TensorDataset

#from sklearn.model_selection import train_test_split

#def main(args):
#%%
#========paths:
#features = "./Features.xlsx"
#ecg_dir = '/vol/hinkelstn/data/'+'ecg_8k_wav'+'/sinus_rhythm_8k/'
#ecg_dir = '/vol/hinkelstn/data/'+'ecg_8k_wav'+'/atrial_fibrillation_8k/'
#rf = "./rf9_nf_cv0_pickle.p"

#========parameters:
verbose = False
#window = 2048 
#stride = 512
sub = 4
#save_name ="flex_2c8,16_2f16_k8_s4_sub4_b512_raw_2K_stable_cv1"
#save_file ="flex_2c8,16_2f16_k8_s4_b512_raw_2K_last-512_nozsc_sub4_meannorm_cv12_qta_float.pth"
#save_file ="train_flex_2c8,16_2f16_k8_s4_b512_raw_2K_last-512_nozsc_sub4_meannorm_cv12_best_prune18.pth"
#model = torch.load(save_file, map_location='cpu')

#model = torch.load('train_'+save_name+'_best.pth', lambda storage, loc: storage.cpu)
model_qta_best  = pickle.load(open('model_qa.pth', 'rb'))
model = torch.quantization.convert(model_qta_best.eval(), inplace=False)


#load_ECG =  torch.load ('raw_x_2K_nofilter_stable.pt') 
load_ECG =  torch.load ('raw_x_2K_nofilter_last-512_nozsc.pt') 


#loaded_vars = pickle.load(open("train_"+save_name+"_variables.p","rb"))


#params = loaded_vars['params']
#epoch = params.epoch
#print("{:>40}  {:<8d}".format("Epoch:", epoch))
seed = 1
#print("{:>40}  {:<8d}".format("Seed:", seed))
test_size = 0.2
np.random.seed(seed)
t_range = range(0,512)

model.eval()

raw_x = load_ECG['raw_x']
data_tag = load_ECG['data_tag']
target = torch.tensor(load_ECG['target'])

raw_x = raw_x[:, :, ::sub]

dataset_splits = create_datasets_win(raw_x, target, data_tag, test_size, seed=seed, t_range = t_range, zero_mean = True)
ecg_datasets = dataset_splits[0:3]
trn_idx, val_idx, tst_idx = dataset_splits[3:6]

batch_size = 512
trn_dl, val_dl, tst_dl = create_loaders(ecg_datasets, bs=batch_size, jobs = 0)
raw_feat = ecg_datasets[0][0][0].shape[0]
raw_size = ecg_datasets[0][0][0].shape[1]
num_classes = 2

batch = []
i_error = []
list_pred = []
y_tst = []
correct, total , total_P, = 0, 0, 0
with torch.no_grad():
    for i_batch, batch in enumerate(tst_dl):
        x_raw, y_batch = [t for t in batch]
        list_x = list(range(i_batch*tst_dl.batch_size,min((i_batch+1)*tst_dl.batch_size,len(tst_dl.dataset))))
       
        out = model(x_raw)
        preds = out.argmax(dim=1)

        list_pred = np.append(list_pred,preds.cpu())
        y_tst = np.append(y_tst,y_batch.cpu())            
        # list_pred = np.append(list_pred,preds.tolist())
        total += y_batch.size(0)
        correct += (preds ==y_batch).sum().item()    
        # i_error = np.append(i_error,np.where(preds !=y_batch))
        i_error = np.append(i_error,[list_x[i] for i in np.where((preds !=y_batch).cpu())[0]])

    
acc = correct / total * 100

TP_ECG, FP_ECG , total_P, total_N = np.zeros(4)
idx_TP = []
idx_FP = []

#    if slide == False:
TP_ECG = ((list_pred == y_tst) & (1 == y_tst)).sum().item()
total_P = (1 ==y_tst).sum().item()
#        TP_ECG_rate = TP / total_P *100
FP_ECG = ((list_pred !=y_tst) & (0 == y_tst)).sum().item() 
total_N = total-total_P

TP_ECG_rate = TP_ECG / total_P *100
FP_ECG_rate = FP_ECG / total_N *100

print("{:>40}  {:<8.3f}".format("TP rate:", TP_ECG_rate))
print("{:>40}  {:<8.3f}".format("FP rate:", FP_ECG_rate))

#%%=================== getting network parameters
y = x_raw[0]
y = y.unsqueeze(0)
outy = model(y)
print(outy)

x = x_raw[0]
x = x.unsqueeze(1)
x = x.unsqueeze(0)
print(x.shape)

print(model)  #print model information
print(model.raw[1].layers[0])  #address model  parts
print(model.raw[1].layers)  #address model  parts
out = model(x_raw)  #compute model output
print(out.shape)

internal1 = torch.nn.Sequential(model.raw[0],model.raw[1].layers[0])(x)    # compute internal variables
print(internal1.shape)

internal2 = torch.nn.Sequential(model.raw[1].layers[1:4],model.raw[2])(internal1)    # compute internal variables
print(internal2.shape)

conv1_1 = model.raw[1].layers[0]  # conv1 in layer 1
print(conv1_1.weight)
print(conv1_1.bias)

#%%=============== batch normalization parameters
in_bn = torch.nn.Sequential(model.raw[0],model.raw[1].layers[0:2])(x)
bn = model.raw[1].layers[2]
gamma = bn.weight
beta = bn.bias
mean = bn.running_mean
var = bn.running_var
eps = bn.eps
var_sqrt = torch.sqrt(var + eps)

#applying batch normalization on channel 3 only
out_bn = ((in_bn[0,3,0,0] - mean[3]) * gamma[3]) / var_sqrt[3]+ beta[3]
out_bn2 = bn(in_bn)
print(out_bn)
print(out_bn2[0,3,0,0])




