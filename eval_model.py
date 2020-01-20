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

from ptflops import get_model_complexity_info
#from thop import profile

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
#%%===============  loading a learned model

save_name = "1d_3conv_2FC_v2_2K_win"
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

save_name2 = input("Input model to load (currently "+save_name+" is selected) :")
if save_name2 != '':
    save_name = save_name2

print(save_name + " is loaded.")

load_ECG =  torch.load ('raw_x_8K_sync_win2K.pt')
#load_ECG =  torch.load ('raw_x_8K_sync.pt') 
#load_ECG =  torch.load ('raw_x_4k_5K.pt') 
#load_ECG =  torch.load ('raw_x_all.pt') 

loaded_vars = pickle.load(open("train_"+save_name+"_variables.p","rb"))
#loaded_file = pickle.load(open("variables"+t_stamp+".p","rb"))
#loaded_file = pickle.load(open("variables_ended"+t_stamp+".p","rb"))

params = loaded_vars['params']
epoch = params.epoch
print('epoch: %d ' % (epoch))
seed = params.seed
test_size = params.test_size
np.random.seed(seed)
t_range = params.t_range

#cuda_num = input("cuda number:")
cuda_num = 0   # export CUDA_VISIBLE_DEVICES=x

device = torch.device('cuda:'+str(cuda_num) if torch.cuda.is_available() and cuda_num != 'cpu' else 'cpu')
#device = torch.device('cpu')

raw_x = load_ECG['raw_x']
#raw_x = load_ECG['raw_x'].to(device)
target = load_ECG['target']
data_tag = load_ECG['data_tag']
#data_tag = load_ECG['IDs']
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

#---------------------  Evaluation function
def evaluate(model, tst_dl, thresh_AF = 3, device = 'cpu'):
    model.to(device)
    s = time.time()
    model.eval()
    correct, total , total_P, = 0, 0, 0
#    TP , FP = 0,0
    
    batch = []
    i_error = []
    list_pred = []
    with torch.no_grad():
        for i_batch, batch in enumerate(tst_dl):
            x_raw, y_batch = [t.to(device) for t in batch]
            list_x = list(range(i_batch*tst_dl.batch_size,min((i_batch+1)*tst_dl.batch_size,len(tst_ds))))
        #    x_raw, y_batch = [t.to(device) for t in batch]
        #    x_raw, y_batch = tst_ds.tensors
            #x_raw, y_batch = [t.to(device) for t in val_ds.tensors]
            out = model(x_raw)
            preds = F.log_softmax(out, dim = 1).argmax(dim=1)
        #    preds = F.log_softmax(out, dim = 1).argmax(dim=1).to('cpu')
            list_pred = np.append(list_pred,preds.tolist())
        #    list_pred = np.append(list_pred,preds.tolist())
            total += y_batch.size(0)
            correct += (preds ==y_batch).sum().item()    
        #    i_error = np.append(i_error,np.where(preds !=y_batch))
            i_error = np.append(i_error,[list_x[i] for i in np.where((preds !=y_batch).to('cpu'))[0]])
        #    TP += ((preds ==y_batch) & (1 ==y_batch)).sum().item()
        #    total_P += (1 ==y_batch).sum().item()
        #    FP += ((preds !=y_batch) & (0 ==y_batch)).sum().item()

    elapsed = time.time() - s
    print('''elapsed time (seconds): {0:.2f}'''.format(elapsed))
        
    acc = correct / total * 100
    #TP_rate = TP / total_P *100
    #FP_rate = FP / (total-total_P) *100
    
    print('Accuracy on all windows of test data:  %2.2f' %(acc))
    
    #TP_rate = TP / (1 ==y_batch).sum().item() *100
    #FP_rate = FP / (0 ==y_batch).sum().item() *100
    
    
    win_size = (data_tag==0).sum()
    # thresh_AF = win_size /2
    # thresh_AF = 3
    
    list_ECG = np.unique([data_tag[i] for i in tst_idx])
    #list_ECG = np.unique([data_tag[i] for i in tst_idx if target[i] == label])
    #len(list_error_ECG)/8000*100
    
    TP_ECG, FP_ECG , total_P, total_N = np.zeros(4)
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
                FP_ECG += 1
                
        
    #TP_ECG_rate = TP_ECG / len(list_ECG) *100
    TP_ECG_rate = TP_ECG / total_P *100
    FP_ECG_rate = FP_ECG / total_N *100
    
    
    print("Threshold for detecting AF: %d" % (thresh_AF))
    print("TP rate: %2.3f" % (TP_ECG_rate))
    print("FP rate: %2.3f" % (FP_ECG_rate))
    
    return TP_ECG_rate, FP_ECG_rate, list_pred_win, elapsed
#print('True positives on test data:  %2.2f' %(TP_rate))
#print('False positives on test data:  %2.2f' %(FP_rate))
#------------------------------------------  

# %%

model = my_net_classes.Classifier_1d_3_conv_2FC_v2(raw_feat, num_classes, raw_size).to(device)
#model = my_net_classes.Classifier_1d_1_conv_1FC(raw_feat, num_classes, raw_size).to(device)
#model = my_net_classes.Classifier_1d_3_conv_2FC(raw_feat, num_classes, raw_size, batch_norm = True, conv_type = '1d').to(device)
#model = my_net_classes.Classifier_1d_6_conv(raw_feat, num_classes, raw_size, batch_norm = True, conv_type = '2d').to(device)
#model = my_net_classes.Classifier_1d_6_conv_ver1(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
#model = my_net_classes.Classifier_1d_6_conv(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
#model = my_net_classes.Classifier_1dconv(raw_feat, num_classes, raw_size).to(device)
#model = my_net_classes.Classifier_1dconv(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
#model = my_net_classes.Classifier_1dconv_BN(raw_feat, num_classes, raw_size, batch_norm = True).to(device)

#if torch.cuda.is_available()*0:
#    model.load_state_dict(torch.load("train_"+save_name+'_best.pth', map_location=lambda storage, loc: storage.cuda('cuda:'+str(cuda_num))))
#else:
model.load_state_dict(torch.load("train_"+save_name+'_best.pth', map_location=lambda storage, loc: storage))


#model = Classifier_1dconv(raw_feat, num_classes, raw_size/(2*4**3)).to(device)
#model.load_state_dict(torch.load("train_"+save_name+'_best.pth'))

thresh_AF = 5

TP_ECG_rate, FP_ECG_rate, list_pred_win, elapsed = evaluate(model, tst_dl, thresh_AF = thresh_AF)

#pickle.dump((TP_ECG_rate, FP_ECG_rate, list_pred_win, elapsed),open(save_name+"result.p","wb"))


#-----------------------  visualize training curve



model2 = model.to('cpu')
summary(model2, input_size=(raw_feat, raw_size), batch_size = batch_size, device = 'cpu')
flops1, params = get_model_complexity_info(model2, (raw_feat, raw_size), as_strings=False, print_per_layer_stat=True);
#%%===============  checking internal values
    
# %%==================================================================== 
# ================================== Quantization
# ====================================================================     
model.to('cpu')


# ----------------- Dynamic Quantization

model_qn = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d} , dtype= torch.qint8
        )

summary(model, input_size=(raw_feat, raw_size), batch_size = batch_size, device = 'cpu')
summary(model_qn, input_size=(raw_feat, raw_size), batch_size = batch_size, device = 'cpu')

#def print_size_of_model(model):
#    torch.save(model.state_dict(), "temp.p")
#    print('Size (MB):', os.path.getsize("temp.p")/1e6)
#    os.remove('temp.p')


#model_qn.to(device)
model_qn.to('cpu');

TP_ECG_rate_q, FP_ECG_rate_q, list_pred_win, elapsed = evaluate(model_qn, tst_dl)


flops1, params = get_model_complexity_info(model, (raw_feat, raw_size), as_strings=False, print_per_layer_stat=True);
flops_q, params_q = get_model_complexity_info(model_qn, (raw_feat, raw_size), as_strings=False, print_per_layer_stat=True);
print('{:<30}  {:<8}'.format('Computational complexity: ', flops_q))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

#input = torch.randn(1, raw_feat, raw_size)
#flops, params = profile(model, inputs=(torch.randn(1, raw_feat, raw_size), ))

#%%===========  Conv2d quantization

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
    
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=(1,1), groups=1):
        super().__init__()
        padding = (0,(kernel_size[1] -1) // 2)
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),            
            nn.Dropout(0.5),
            nn.ReLU(inplace=False)
#            nn.Dropout(0.5)
        )

class Flatten2(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.reshape(x.size(0), -1)
        return x.reshape(-1)        

class dumy_CNN(nn.Module):
    def __init__(self, ni, no, raw_size):
        super().__init__()        
        
        self.layers = nn.Sequential(       
#        self.conv = nn.Conv2d(ni, no, (1,8),bias = False)
#        self.convB = ConvBNReLU(ni,no,(1,9))
        ConvBNReLU(ni,no,(1,9)),
        Flatten2(),
#        my_net_classes.Flatten(),
        nn.Linear(raw_size*no, 2)
        )
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)        
       
    def forward(self, x):
        out = self.quant(x)
        out  =  out.unsqueeze(2)
        out  = self.layers(out)
        out  = self.dequant(out)
        return out    

def evaluation1(model_test,tst_dl):
    correct, total = 0, 0
    with torch.no_grad():
        for i_batch, batch in enumerate(tst_dl):
            x_raw, y_batch = [t.to('cpu') for t in batch]
            out = model_test(x_raw)
            preds = F.log_softmax(out, dim = 1).argmax(dim=1)    
            total += y_batch.size(0)
            correct += (preds ==y_batch).sum().item()    
            
    acc = correct / total * 100
    print(acc)    
    return(acc)
    
model_test = dumy_CNN(2,10, 2048)
model_test.eval()

model_test.fuse_model()

model_test.qconfig = torch.quantization.default_qconfig

print(model_test.qconfig)
torch.quantization.prepare(model_test, inplace=True)

# Calibrate with the training set
evaluation1(model_test,tst_dl)
print('Post Training Quantization: Calibration done')

# Convert to quantized model
model_qn = torch.quantization.convert(model_test)
print('Post Training Quantization: Convert done')


evaluation1(model_qn,tst_dl)

#model_qn = torch.quantization.quantize_dynamic(
#        model_test, {nn.Linear, nn.Conv2d} , dtype= torch.qint8
#        )
#print(model_qn)
model_qn.conv.weight.data[0,0,0,0].item()


summary(model_test, input_size=(raw_feat, raw_size), batch_size = batch_size, device = 'cpu')
summary(model_qn, input_size=(raw_feat, raw_size), batch_size = batch_size, device = 'cpu')

flops, params = get_model_complexity_info(model_test, (raw_feat, raw_size), as_strings=False, print_per_layer_stat=True);
flops_q, params_q = get_model_complexity_info(model_qn, (raw_feat, raw_size), as_strings=False, print_per_layer_stat=True);

#input = torch.randn(1, raw_feat, raw_size)
#flops, params = profile(model_test, inputs=(torch.randn(1, raw_feat, raw_size), ))

#%%===========  static quantization
num_calibration_batches = 10
model.to('cpu')
model.eval()



