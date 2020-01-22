#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:53:16 2019

@author: bhossein
"""

#import torch
import torch
from torch import nn
import numpy as np
from torch.quantization import QuantStub, DeQuantStub
import time
from torch.nn import functional as F

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
#        return self.pointwise(x)

#%% ==================  2D output            
class _SepConv1d_2d(nn.Module):
    """  simple separable convolution implementation based on Conv2d module.
    
    The separable convlution is a method to reduce number of the parameters 
    in the deep learning network for slight decrease in predictions quality.
    """
    
    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()        
        self.depthwise = nn.Conv2d(ni, ni, kernel_size = (1,kernel), stride = (1,stride), padding = (0,pad), groups = ni)
        self.pointwise = nn.Conv2d(ni, no, kernel_size=(1,1))
        
    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x.squeeze()
#        return x.reshape(x.size(0),x.size(1),x.size(3))
#        return x
#         return self.pointwise(self.depthwise(x.unsqueeze(2)))

#%% ==================   3D output 
class _SepConv1d_2d_v2(nn.Module):
    """  simple separable convolution implementation based on Conv2d module.
    
    The separable convlution is a method to reduce number of the parameters 
    in the deep learning network for slight decrease in predictions quality.
    """
    
    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv2d(ni, ni, kernel_size = (1,kernel), stride = (1,stride), padding = (0,pad), groups = ni)
        self.pointwise = nn.Conv2d(ni, no, kernel_size=(1,1))
        
    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.depthwise(x)
        x = self.pointwise(x)
#        return x.squeeze()
#        return x.reshape(x.size(0),x.size(1),x.size(3))
        return x
#         return self.pointwise(self.depthwise(x.unsqueeze(2)))        

#%% ==================   Version 1      
class SepConv1d_ver1(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    
    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride, pad, drop=None, batch_norm = None,
                 activ=lambda: nn.ReLU(inplace=True)):
    
        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        
#        if drop is not None:
#            layers.append(nn.Dropout(drop))      
        
        if activ:
            layers.append(activ())
        
        if batch_norm:
            layers.append(nn.BatchNorm1d(num_features = no))
        if drop is not None:
            layers.append(nn.Dropout(drop))
            
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.layers(x)
    
#%% ==================          SepConv1d
class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    
    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride, pad, drop=None, batch_norm = None,
                 conv_type = '1d', activ=lambda: nn.ReLU(inplace=True)):
    
        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        
        if conv_type =='1d':
            layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        elif conv_type =='2d':
            layers = [_SepConv1d_2d(ni, no, kernel, stride, pad)]
        
#        if drop is not None:
#            layers.append(nn.Dropout(drop))      
        
        if batch_norm:
            layers.append(nn.BatchNorm1d(num_features = no))

        if drop is not None:
            layers.append(nn.Dropout(drop))
            
        if activ:
            layers.append(activ())        
                
#        if drop is not None:
#            layers.append(nn.Dropout(drop))
            
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.layers(x)    
    
#%% ==================          SepConv1d  -  change: dropouts after activation
class SepConv1d_v2(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    
    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride, pad, drop=0.2, batch_norm = None,
                 conv_type = '1d', activ=lambda: nn.ReLU(inplace=True)):
    
        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        
        if conv_type =='1d':
            layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        elif conv_type =='2d':
            layers = [_SepConv1d_2d(ni, no, kernel, stride, pad)]
        
#        if drop is not None:
#            layers.append(nn.Dropout(drop))      
        
        if batch_norm:
            layers.append(nn.BatchNorm1d(num_features = no))        
            
        if activ:
            layers.append(activ())        
            
        if drop is not None:
            layers.append(nn.Dropout(drop))            
                
#        if drop is not None:
#            layers.append(nn.Dropout(drop))
            
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.layers(x)   
#%% ==================        SepConv1d  -  change: batch2d
class SepConv1d_v3(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    
    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride, pad, drop=0.2, batch_norm = None,
                 conv_type = '1d', activ=lambda: nn.ReLU(inplace=True)):
    
        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        
        if conv_type =='1d':
            layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        elif conv_type =='2d':
            layers = [_SepConv1d_2d_v2(ni, no, kernel, stride, pad)]
        
#        if drop is not None:
#            layers.append(nn.Dropout(drop))      
        
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features = no))        
            
        if activ:
            layers.append(activ())        
            
        if drop is not None:
            layers.append(nn.Dropout(drop))            
                
#        if drop is not None:
#            layers.append(nn.Dropout(drop))
            
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.layers(x).squeeze()

#%% ==================        SepConv1d  -  sep conv and BN and drp together
class SepConv1d_v4(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    
    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride, pad, drop=0.2, batch_norm = None,
                 conv_type = '1d', activ=lambda: nn.ReLU(inplace=True)):
    
        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        
        layers = [nn.Conv2d(ni, ni, kernel_size = (1,kernel), stride = (1,stride), padding = (0,pad), groups = ni)]
        layers.append(nn.Conv2d(ni, no, kernel_size=(1,1)))
        
#        if drop is not None:
#            layers.append(nn.Dropout(drop))      
        
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features = no))        
            
        if activ:
            layers.append(activ())        
            
        if drop is not None:
            layers.append(nn.Dropout(drop))            
                
            
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x): 
#        x = torch.unsqueeze(x,2)
        x = self.layers(x)
#        x = x.squeeze()
        return x
        
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
#%% ==================         FLAtten,  change: reshape instead of view
class Flatten2(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.reshape(x.size(0), -1)
        return x.reshape(-1)    
#%% ==================         
class parameters():
    """
    saves the training parameters  
    """
    def __init__(self, lr, epoch, patience, step, batch_size, t_range, seed, test_size):
        self.lr = lr    
        self.epoch = epoch                
        self.patience = patience
        self.step = step
        self.batch_size = batch_size
        self.t_range = t_range
        self.seed = seed
        self.test_size = test_size

#%%==================  Evaluation function
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
   
#%% ==================         
#class Classifier(nn.Module):
#    def __init__(self, raw_ni, no, drop=.5):
#        super().__init__()
#        
#        self.raw = nn.Sequential(
#            SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop),
#            SepConv1d(    32,  64, 8, 4, 2, drop=drop),
#            SepConv1d(    64, 128, 8, 4, 2, drop=drop),
#            SepConv1d(   128, 256, 8, 4, 2),
#            Flatten(),
#            nn.Dropout(drop), nn.Linear(256, 64), nn.ReLU(inplace=True),
#            nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))
#        
##        self.fft = nn.Sequential(
##            SepConv1d(fft_ni,  32, 8, 2, 4, drop=drop),
##            SepConv1d(    32,  64, 8, 2, 4, drop=drop),
##            SepConv1d(    64, 128, 8, 4, 4, drop=drop),
##            SepConv1d(   128, 128, 8, 4, 4, drop=drop),
##            SepConv1d(   128, 256, 8, 2, 3),
##            Flatten(),
##            nn.Dropout(drop), nn.Linear(256, 64), nn.ReLU(inplace=True),
##            nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))
#        
#        self.out = nn.Sequential(
##            nn.Linear(128, 64), nn.ReLU(inplace=True), 
#            nn.Linear(64, no))
#        
#    def forward(self, t_raw):
#        raw_out = self.raw(t_raw)
##        fft_out = self.fft(t_fft)
##        t_in = torch.cat([raw_out, fft_out], dim=1)
#        out = self.out(raw_out)
#        return out
# %%===============================  1dconv - 1 Batch-normal 
class Classifier_1dconv_BN(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
        super().__init__()
        
#        assert int(n_flt) == n_flt
        flat_in = 256 * int (raw_size / (2*4*4*4))
#        assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
#        flat_in = 256*int(n_flt)
        
        
        if batch_norm:
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop),
                SepConv1d(   128, 256, 8, 4, 2),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 64), nn.ReLU(inplace=True),
                nn.BatchNorm1d(num_features = 64),
                nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True),            
                nn.BatchNorm1d(num_features = 64))
        else:        
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop),
                SepConv1d(    32,  64, 8, 4, 2, drop=drop),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop),
                SepConv1d(   128, 256, 8, 4, 2),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 64), nn.ReLU(inplace=True),    
                nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))

        
        self.out = nn.Sequential(
#            nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(64, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
#        fft_out = self.fft(t_fft)
#        t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(raw_out)
        return out    
    
#%% ==================   1dconv - 4 conv
class Classifier_1dconv(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
        super().__init__()
        
#        assert int(n_flt) == n_flt
        flat_in = 256 * int (raw_size / (2*4*4*4))
#        assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
#        flat_in = 256*int(n_flt)
        
        
        if batch_norm:
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   128, 256, 8, 4, 2, batch_norm = batch_norm),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 64), nn.ReLU(inplace=True),
                nn.BatchNorm1d(num_features = 64),
                nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True),            
                nn.BatchNorm1d(num_features = 64))
        else:        
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop),
                SepConv1d(    32,  64, 8, 4, 2, drop=drop),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop),
                SepConv1d(   128, 256, 8, 4, 2),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 64), nn.ReLU(inplace=True),    
                nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))


        
        self.out = nn.Sequential(
#            nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(64, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
#        fft_out = self.fft(t_fft)
#        t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(raw_out)
        return out    
    

#%% ==================   1dconv - 6 conv - 3 FC  : VER 1
class Classifier_1d_6_conv_ver1(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True):
        super().__init__()
        
#        assert int(n_flt) == n_flt
        flat_in = 1024 * int (raw_size / (2*4*4*4*4*4))
#        assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
#        flat_in = 256*int(n_flt)
        
        
        if batch_norm:
            self.raw = nn.Sequential(
                SepConv1d_ver1(raw_ni,  32, 8, 2, 3, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
                SepConv1d_ver1(    32,  64, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d_ver1(    64, 128, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d_ver1(   128, 256, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d_ver1(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d_ver1(   512,1024, 8, 4, 2, batch_norm = batch_norm),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 128), nn.ReLU(inplace=True),
                nn.BatchNorm1d(num_features = 128),
                nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True),            
                nn.BatchNorm1d(num_features = 128))
        else:        
            self.raw = nn.Sequential(
                SepConv1d_ver1(raw_ni,  32, 8, 2, 3, drop=drop),  #out: raw_size/str
                SepConv1d_ver1(    32,  64, 8, 4, 2, drop=drop),
                SepConv1d_ver1(    64, 128, 8, 4, 2, drop=drop),
                SepConv1d_ver1(   128, 256, 8, 4, 2, drop=drop),
                SepConv1d_ver1(   256, 512, 8, 4, 2, drop=drop),
                SepConv1d_ver1(   512,1024, 8, 4, 2),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 128), nn.ReLU(inplace=True),                
                nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True))
            
        
        self.out = nn.Sequential(
#            nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
#        fft_out = self.fft(t_fft)
#        t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(raw_out)
        return out    
    
#%% ==================   1dconv - 6 conv - 3 FC
class Classifier_1d_6_conv(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True, conv_type = '2d'):
        super().__init__()
        
#        assert int(n_flt) == n_flt
        flat_in = 1024 * int (raw_size / (2*4*4*4*4*4))
#        assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
#        flat_in = 256*int(n_flt)
        
        
        if batch_norm:
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 2, 3, drop, batch_norm, conv_type),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop, batch_norm, conv_type),
                SepConv1d(    64, 128, 8, 4, 2, drop, batch_norm, conv_type),
                SepConv1d(   128, 256, 8, 4, 2, drop, batch_norm, conv_type),
                SepConv1d(   256, 512, 8, 4, 2, drop, batch_norm, conv_type),
                SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm, conv_type = conv_type),
                Flatten(),
                nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True),       
                nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))
            
#                nn.Dropout(drop), nn.Linear(flat_in, 128), nn.ReLU(inplace=True),
#                nn.BatchNorm1d(num_features = 128),
#                nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True),            
#                nn.BatchNorm1d(num_features = 128))
        else:        
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop),
                SepConv1d(   512,1024, 8, 4, 2),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 128), nn.ReLU(inplace=True),                
                nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True))
            
        
        self.out = nn.Sequential(
#            nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
#        fft_out = self.fft(t_fft)
#        t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(raw_out)
        return out    

#%% ==================   1dconv - 6 conv - 3 FC   drop after relu + BN2d
class Classifier_1d_6_conv_v2(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True, conv_type = '2d'):
        super().__init__()
        
#        assert int(n_flt) == n_flt
#        flat_in = 1024 * int (raw_size / (2*4*4*4*4*4))
        flat_in = 1024 * int (raw_size / (2*4*4*4*4*4))
#        assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
#        flat_in = 256*int(n_flt)
        
        

        self.raw = nn.Sequential(
            SepConv1d_v4(raw_ni,  32, 8, 4, 3, drop, batch_norm, conv_type),  #out: raw_size/str
#            ConvBNReLU(raw_ni,32,(1,9)),
            SepConv1d_v4(    32,  64, 8, 4, 2, drop, batch_norm, conv_type),
            SepConv1d_v4(    64, 128, 8, 4, 2, drop, batch_norm, conv_type),
            SepConv1d_v4(   128, 256, 8, 4, 2, drop, batch_norm, conv_type),
#            SepConv1d_v4(   256, 512, 8, 4, 2, drop, batch_norm, conv_type),
#            SepConv1d_v4(   512,1024, 8, 4, 2, batch_norm = batch_norm, conv_type = conv_type)            
            )

        self.FC = nn.Sequential(
            Flatten(),
#            Flatten2(),
            nn.Linear(flat_in, 128), nn.ReLU(inplace=True), nn.Dropout(drop),
#            nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128),  nn.ReLU(inplace=True), nn.Dropout(drop),            
            nn.Linear( 128, 128),    nn.ReLU(inplace=True), nn.Dropout(drop)
#            nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.ReLU(inplace=True), nn.Dropout(drop)
                )
                            
        self.out = nn.Sequential(
#            nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(128, no))
#            nn.Linear(1024, no))
        
        self.quant = QuantStub()
        
        self.dequant = DeQuantStub()
        
    def fuse_model2(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['1', '2','3'], inplace=True)
#                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
    
    def fuse_model(self):
        for m in self.modules():
            if type(m) == SepConv1d_v4:
                fuse_profile = ['layers.1', 'layers.2', 'layers.3']
#                fuse_profile = ['layers.0.pointwise', 'layers.1', 'layers.2']
                torch.quantization.fuse_modules(m, fuse_profile, inplace=True)
#                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
        torch.quantization.fuse_modules(self.FC, [['1','2'],['4','5']], inplace=True)
                
                
    def forward(self, t_raw):
        t_raw = self.quant(t_raw)        
        t_raw  =  t_raw.unsqueeze(2)
        raw_out = self.raw(t_raw)
#        fft_out = self.fft(t_fft)
#        t_in = torch.cat([raw_out, fft_out], dim=1)
        FC_out = self.FC(raw_out)        
        out = self.out(FC_out)
        
        out  = self.dequant(out)
        return out    


#%% ==================   1dconv - 5 conv - 2 FC   drop after relu + BN2d
class Classifier_1d_5_conv_v2(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True, conv_type = '2d'):
        super().__init__()
        
#        assert int(n_flt) == n_flt
#        flat_in = 1024 * int (raw_size / (2*4*4*4*4*4))
        flat_in = 512 * int (raw_size / (4*4*4*4*4))
#        assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
#        flat_in = 256*int(n_flt)
        
        

        self.raw = nn.Sequential(
            SepConv1d_v4(raw_ni,  32, 8, 4, 3, drop, batch_norm, conv_type),  #out: raw_size/str
#            ConvBNReLU(raw_ni,32,(1,9)),
            SepConv1d_v4(    32,  64, 8, 4, 2, drop, batch_norm, conv_type),
            SepConv1d_v4(    64, 128, 8, 4, 2, drop, batch_norm, conv_type),
            SepConv1d_v4(   128, 256, 8, 4, 2, drop, batch_norm, conv_type),
            SepConv1d_v4(   256, 512, 2, 4, 0, drop, batch_norm, conv_type),
#            SepConv1d_v4(   512,1024, 2, 4, 2, batch_norm = batch_norm, conv_type = conv_type)            
            )

        self.FC = nn.Sequential(
            Flatten2(),
#            nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128),  nn.ReLU(inplace=True), nn.Dropout(drop)
            nn.Linear(flat_in, 128), nn.ReLU(inplace=True), nn.Dropout(drop)
#            nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True)                
                )
                            
        self.out = nn.Sequential(
#            nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(128, no))
#            nn.Linear(1024, no))
        
        self.quant = QuantStub()
        
        self.dequant = DeQuantStub()
        
    def fuse_model2(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['1', '2','3'], inplace=True)
#                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
    
    def fuse_model(self):
        for m in self.modules():
            if type(m) == SepConv1d_v4:
                fuse_profile = ['layers.1', 'layers.2', 'layers.3']
#                fuse_profile = ['layers.0.pointwise', 'layers.1', 'layers.2']
                torch.quantization.fuse_modules(m, fuse_profile, inplace=True)
#                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
        torch.quantization.fuse_modules(self.FC, ['1','2'], inplace=True)
                
                
    def forward(self, t_raw):
        t_raw = self.quant(t_raw)        
        t_raw  =  t_raw.unsqueeze(2)
        raw_out = self.raw(t_raw)
#        fft_out = self.fft(t_fft)
#        t_in = torch.cat([raw_out, fft_out], dim=1)
        FC_out = self.FC(raw_out)        
        out = self.out(FC_out)
        
        out  = self.dequant(out)
        return out    
#%% ==================   1dconv - 3 conv - 2 FC
class Classifier_1d_3_conv_2FC(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True, conv_type = '2d'):
        super().__init__()
        
#        assert int(n_flt) == n_flt
        flat_in = 128 * int (raw_size / (2*4*4))
#        assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
#        flat_in = 256*int(n_flt)
        
        
#        if batch_norm:
        self.raw = nn.Sequential(
            SepConv1d(raw_ni,  32, 8, 2, 3, drop, batch_norm, conv_type),  #out: raw_size/str
            SepConv1d(    32,  64, 8, 4, 2, drop, batch_norm, conv_type),
            SepConv1d(    64, 128, 8, 4, 2, drop, batch_norm, conv_type),
#                SepConv1d(   128, 256, 8, 4, 2, drop, batch_norm, conv_type),
#                SepConv1d(   256, 512, 8, 4, 2, drop, batch_norm, conv_type),
#                SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm, conv_type = conv_type),
            Flatten(),
            nn.Linear(flat_in, 512), nn.BatchNorm1d(num_features = 512), nn.Dropout(drop), nn.ReLU(inplace=True)
#                nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True)
            )            
        
        self.out = nn.Sequential(
#            nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(512, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
#        fft_out = self.fft(t_fft)
#        t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(raw_out)
        return out    

#%% ==================   1dconv - 3 conv - 2 FC   -  change: dropouts after activation
class Classifier_1d_3_conv_2FC_v2(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True, conv_type = '2d'):
        super().__init__()

        drop0 = 0.2
#        assert int(n_flt) == n_flt
        flat_in = 128 * int (raw_size / (2*4*4))
#        assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
#        flat_in = 256*int(n_flt)
        
        
#        if batch_norm:
        self.raw = nn.Sequential(
            SepConv1d_v2(raw_ni,  32, 8, 2, 3, drop0, batch_norm, conv_type),  #out: raw_size/str
            SepConv1d_v2(    32,  64, 8, 4, 2, drop0, batch_norm, conv_type),
            SepConv1d_v2(    64, 128, 8, 4, 2, drop0, batch_norm, conv_type),
#                SepConv1d(   128, 256, 8, 4, 2, drop, batch_norm, conv_type),
#                SepConv1d(   256, 512, 8, 4, 2, drop, batch_norm, conv_type),
#                SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm, conv_type = conv_type),
            Flatten(),
            nn.Linear(flat_in, 512), nn.BatchNorm1d(num_features = 512), nn.Dropout(drop), nn.ReLU(inplace=True)
#                nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True)
            )            
        
        self.out = nn.Sequential(
#            nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(512, no))
        
    def forward(self, t_raw):        
#        t_raw = self.quant(t_raw)        
        raw_out = self.raw(t_raw)
#        fft_out = self.fft(t_fft)
#        t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(raw_out)
#        out  = self.dequant(out)
        return out    
    
#%% ==================   1dconv - 1 conv - 1 FC
class Classifier_1d_1_conv_1FC(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True, conv_type = '2d'):
        super().__init__()
        
#        assert int(n_flt) == n_flt
        flat_in = 32 * int (raw_size / (4))
#        assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
#        flat_in = 256*int(n_flt)
        
        
#        if batch_norm:
        self.raw = nn.Sequential(
            SepConv1d(raw_ni,  32, 8, 4, 3, drop, batch_norm, conv_type),  #out: raw_size/str
#            SepConv1d(    32,  64, 8, 4, 2, drop, batch_norm, conv_type),
#            SepConv1d(    64, 128, 8, 4, 2, drop, batch_norm, conv_type),
#                SepConv1d(   128, 256, 8, 4, 2, drop, batch_norm, conv_type),
#                SepConv1d(   256, 512, 8, 4, 2, drop, batch_norm, conv_type),
#                SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm, conv_type = conv_type),
            Flatten(),
            nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True)
#                nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True)
            )            
        
        self.out = nn.Sequential(
#            nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(128, no))
        
    def forward(self, x):
        x = self.raw(x)
#        fft_out = self.fft(t_fft)
#        t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(x)
        return out        

#%% ==================   dummy net        
class dumy_CNN(nn.Module):
    def __init__(self, ni, no):
        super().__init__()
        
#        flat_in = int ((raw_size / 2)*10)
        
        self.conv = nn.Conv2d(ni, no, 8, 2, 3)
        self.lin = nn.Linear(1024, 4)
        
#        self.layers = nn.Sequential(
#                nn.Conv2d(ni,  10, 8, 2, 3),
#                nn.ReLU(inplace=True),
#                Flatten(),
#                nn.Linear(1024, 128),
#                nn.ReLU(inplace=True))
        
#        self.out = nn.Sequential(
#            nn.Linear(128, no))
        
    def forward(self, x):
#        x_out = self.layers(x)
#        fft_out = self.fft(t_fft)
#        t_in = torch.cat([raw_out, fft_out], dim=1)
#        out = self.out(x_out)
        out = self.lin(self.conv(x))
        return out    

#=====================

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=(1,1), groups=1):
        super().__init__()
        padding = (0,(kernel_size[1] -1) // 2)
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.Conv2d(out_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),            
#            nn.Dropout(0.5),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5)
        )