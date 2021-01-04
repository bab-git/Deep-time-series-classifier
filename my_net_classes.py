#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:53:16 2019

@author: bhossein
"""

import torch
from torch import nn
from torch.quantization import QuantStub, DeQuantStub
import numpy as np

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
        return self.pointwise(self.depthwise(x.unsqueeze(2))).squeeze()

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
        
        # if drop is not None:
        #     layers.append(nn.Dropout(drop))      
        
        if activ:
            layers.append(activ())
        
        if batch_norm:
            layers.append(nn.BatchNorm1d(num_features = no))
        if drop is not None:
            layers.append(nn.Dropout(drop))
            
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.layers(x)
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
#%% ==================    SepConv1d  -  change: add relu_ instead of relu for inplace
class SepConv1d_v5(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    
    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride, pad, drop=0.2, batch_norm = None,
                 conv_type = '1d', activ=lambda: nn.ReLU()):
    
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
#%% ==================    SepConv1d  -  change: no bias
class SepConv1d_v6(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    
    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride, pad, drop=0.2, batch_norm = None,
                 conv_type = '1d', activ=lambda: nn.ReLU(), bias = True):
    
        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        
        layers = [nn.Conv2d(ni, ni, kernel_size = (1,kernel), stride = (1,stride), padding = (0,pad), groups = ni, bias = bias)]
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
        
    #    if drop is not None:
        #    layers.append(nn.Dropout(drop))      
        
        if batch_norm:
            layers.append(nn.BatchNorm1d(num_features = no))

        if drop is not None:
            layers.append(nn.Dropout(drop))
            
        if activ:
            layers.append(activ())        
                
    #    if drop is not None:
        #    layers.append(nn.Dropout(drop))
            
            
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
    def __init__(self, lr, batch_size, t_range, seed, test_size, use_norm=False, zero_mean=False, sub=None, cv_splits=0, cv_repeats=0, epoch=0, patience=0, step=0):
        self.lr = lr    
        self.epoch = epoch                
        self.patience = patience
        self.step = step
        self.batch_size = batch_size
        self.t_range = t_range
        self.seed = seed
        self.test_size = test_size
        self.use_norm = use_norm
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        self.zero_mean = zero_mean
        self.sub = sub

       
   
#%% ==================         
class Classifier(nn.Module):
    def __init__(self, raw_ni, no, drop=.5):
        super().__init__()
       
        self.raw = nn.Sequential(
            SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop),
            SepConv1d(    32,  64, 8, 4, 2, drop=drop),
            SepConv1d(    64, 128, 8, 4, 2, drop=drop),
            SepConv1d(   128, 256, 8, 4, 2),
            Flatten(),
            nn.Dropout(drop), nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))
       
        # self.fft = nn.Sequential(
        #     SepConv1d(fft_ni,  32, 8, 2, 4, drop=drop),
        #     SepConv1d(    32,  64, 8, 2, 4, drop=drop),
        #     SepConv1d(    64, 128, 8, 4, 4, drop=drop),
        #     SepConv1d(   128, 128, 8, 4, 4, drop=drop),
        #     SepConv1d(   128, 256, 8, 2, 3),
        #     Flatten(),
        #     nn.Dropout(drop), nn.Linear(256, 64), nn.ReLU(inplace=True),
        #     nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))
       
        self.out = nn.Sequential(
            # nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(64, no))
       
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        # fft_out = self.fft(t_fft)
        # t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(raw_out)
        return out
# %%===============================  1dconv - 1 Batch-normal 
class Classifier_1dconv_BN(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
        super().__init__()
        
    #    assert int(n_flt) == n_flt
        flat_in = 256 * int (raw_size / (2*4*4*4))
    #    assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
    #    flat_in = 256*int(n_flt)
        
        
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
        #    nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(64, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
    #    fft_out = self.fft(t_fft)
    #    t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(raw_out)
        return out    
#%% ==================   1dconv - 4 conv
class Classifier_1dconv(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
        super().__init__()
        
        # assert int(n_flt) == n_flt
        flat_in = 256 * int (raw_size / (2*4*4*4))
        # assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
        # flat_in = 256*int(n_flt)
        
        
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
            # nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(64, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        # fft_out = self.fft(t_fft)
        # t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(raw_out)
        return out    
#%% ==================   1dconv - 6 conv - 3 FC  : VER 1
class Classifier_1d_6_conv_ver1(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
        super().__init__()
        
        # assert int(n_flt) == n_flt
        flat_in = 1024 * int (raw_size / (2*4*4*4*4*4))
        # assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
        # flat_in = 256*int(n_flt)
        
        
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
            # nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        # fft_out = self.fft(t_fft)
        # t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(raw_out)
        return out    
#%% ==================   1dconv - 6 conv - 3 FC
class Classifier_1d_6_conv(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None, conv_type = '1d'):
        super().__init__()
        
        # assert int(n_flt) == n_flt
        flat_in = 1024 * int (raw_size / (2*4*4*4*4*4))
        # assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
        # flat_in = 256*int(n_flt)
        
        
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
            
                # nn.Dropout(drop), nn.Linear(flat_in, 128), nn.ReLU(inplace=True),
                # nn.BatchNorm1d(num_features = 128),
                # nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True),            
                # nn.BatchNorm1d(num_features = 128))
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
            # nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        # fft_out = self.fft(t_fft)
        # t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(raw_out)
        return out

#%% ==================   1dconv - 3 conv - 2 FC
class Classifier_1d_3_conv_2FC(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True, conv_type = '2d'):
        super().__init__()
        
        # assert int(n_flt) == n_flt
        flat_in = 128 * int (raw_size / (2*4*4))
        # assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
        # flat_in = 256*int(n_flt)
        
        
        # if batch_norm:
        self.raw = nn.Sequential(
            SepConv1d(raw_ni,  32, 8, 2, 3, drop, batch_norm, conv_type),  #out: raw_size/str
            SepConv1d(    32,  64, 8, 4, 2, drop, batch_norm, conv_type),
            SepConv1d(    64, 128, 8, 4, 2, drop, batch_norm, conv_type),
                # SepConv1d(   128, 256, 8, 4, 2, drop, batch_norm, conv_type),
                # SepConv1d(   256, 512, 8, 4, 2, drop, batch_norm, conv_type),
                # SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm, conv_type = conv_type),
            Flatten(),
            nn.Linear(flat_in, 512), nn.BatchNorm1d(num_features = 512), nn.Dropout(drop), nn.ReLU(inplace=True)
                # nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True)
            )            
        
        self.out = nn.Sequential(
            # nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(512, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        # fft_out = self.fft(t_fft)
        # t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(raw_out)
        return out    

#%% ==================   1dconv - 1 conv - 1 FC
class Classifier_1d_1_conv_1FC(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True, conv_type = '2d'):
        super().__init__()
        
        # assert int(n_flt) == n_flt
        flat_in = 32 * int (raw_size / (4))
        # assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
        # flat_in = 256*int(n_flt)
        
        
        # if batch_norm:
        self.raw = nn.Sequential(
            SepConv1d(raw_ni,  32, 8, 4, 3, drop, batch_norm, conv_type),  #out: raw_size/str
            # SepConv1d(    32,  64, 8, 4, 2, drop, batch_norm, conv_type),
            # SepConv1d(    64, 128, 8, 4, 2, drop, batch_norm, conv_type),
            #     SepConv1d(   128, 256, 8, 4, 2, drop, batch_norm, conv_type),
            #     SepConv1d(   256, 512, 8, 4, 2, drop, batch_norm, conv_type),
            #     SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm, conv_type = conv_type),
            Flatten(),
            nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True)
                # nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True)
            )            
        
        self.out = nn.Sequential(
            # nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(128, no))
        
    def forward(self, x):
        x = self.raw(x)
        # fft_out = self.fft(t_fft)
        # t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(x)
        return out        

#%% ==================   dummy net        
class dumy_CNN(nn.Module):
    def __init__(self, ni, no):
        super().__init__()
        
        # flat_in = int ((raw_size / 2)*10)
        
        self.conv = nn.Conv2d(ni, no, 8, 2, 3)
        self.lin = nn.Linear(1024, 4)
        
        # self.layers = nn.Sequential(
        #         nn.Conv2d(ni,  10, 8, 2, 3),
        #         nn.ReLU(inplace=True),
        #         Flatten(),
        #         nn.Linear(1024, 128),
        #         nn.ReLU(inplace=True))
            
        # self.out = nn.Sequential(
        #     nn.Linear(128, no))
        
    def forward(self, x):
        # x_out = self.layers(x)
        # fft_out = self.fft(t_fft)
        # t_in = torch.cat([raw_out, fft_out], dim=1)
        # out = self.out(x_out)
        out = self.lin(self.conv(x))
        return out    
#%% ==================   1dconv - 6 conv - 3 FC - BN on conv - dropout on FC
class Classifier_1d_6_conv_nodropbatch(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
        super().__init__()
        
        # assert int(n_flt) == n_flt
        flat_in = 1024 * int (raw_size / (2*4*4*4*4*4))
        # assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
        # flat_in = 256*int(n_flt)
        
        
        if batch_norm:
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 2, 3, drop=None, batch_norm = batch_norm),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=None, batch_norm = batch_norm),
                SepConv1d(    64, 128, 8, 4, 2, drop=None, batch_norm = batch_norm),
                SepConv1d(   128, 256, 8, 4, 2, drop=None, batch_norm = batch_norm),
                SepConv1d(   256, 512, 8, 4, 2, drop=None, batch_norm = batch_norm),
                SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
                Flatten(),
                nn.Linear(flat_in, 128), nn.Dropout(drop), nn.ReLU(inplace=True),       
                nn.Linear( 128, 128),    nn.Dropout(drop), nn.ReLU(inplace=True))
        else:        
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 2, 3, drop=None),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=None),
                SepConv1d(    64, 128, 8, 4, 2, drop=None),
                SepConv1d(   128, 256, 8, 4, 2, drop=None),
                SepConv1d(   256, 512, 8, 4, 2, drop=None),
                SepConv1d(   512,1024, 8, 4, 2),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 128), nn.ReLU(inplace=True),                
                nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            # nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        # fft_out = self.fft(t_fft)
        # t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(raw_out)
        return out    
#%% ==================   1dconv - 6 conv - 2 FC
class Classifier_1d_6_conv_2_fc(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
        super().__init__()
        
        # assert int(n_flt) == n_flt
        flat_in = 1024 * int (raw_size / (2*4*4*4*4*4))
        # assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
        # flat_in = 256*int(n_flt)
        
        
        if batch_norm:
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
                Flatten(),
                nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))    
                # nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))
        
                # nn.Dropout(drop), nn.Linear(flat_in, 128), nn.ReLU(inplace=True),
                # nn.BatchNorm1d(num_features = 128),
                # nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True),            
                # nn.BatchNorm1d(num_features = 128))
        else:        
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop),
                SepConv1d(   512,1024, 8, 4, 2),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 128), nn.ReLU(inplace=True))              
                # nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True))
            
        # self.fft = nn.Sequential(
        #     SepConv1d(fft_ni,  32, 8, 2, 4, drop=drop),
        #     SepConv1d(    32,  64, 8, 2, 4, drop=drop),
        #     SepConv1d(    64, 128, 8, 4, 4, drop=drop),
        #     SepConv1d(   128, 128, 8, 4, 4, drop=drop),
        #     SepConv1d(   128, 256, 8, 2, 3),
        #     Flatten(),
        #     nn.Dropout(drop), nn.Linear(256, 64), nn.ReLU(inplace=True),
        #     nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            # nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        # fft_out = self.fft(t_fft)
        # t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(raw_out)
        return out    
#%% ==================   1dconv - 5 conv - 1 pool - 3 FC
class Classifier_1d_5_conv_1_pool(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
        super().__init__()
        
        # assert int(n_flt) == n_flt
        flat_in = 512 * int (raw_size / (2*4*4*4*4*2))
        # assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
        # flat_in = 256*int(n_flt)
        
        
        if batch_norm:
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                # SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
                nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True),       
                nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))
            
                # nn.Dropout(drop), nn.Linear(flat_in, 128), nn.ReLU(inplace=True),
                # nn.BatchNorm1d(num_features = 128),
                # nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True),            
                # nn.BatchNorm1d(num_features = 128))
        else:        
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop),
                # SepConv1d(   512,1024, 8, 4, 2),
                nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 128), nn.ReLU(inplace=True),                
                nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True))
            
        # self.fft = nn.Sequential(
        #     SepConv1d(fft_ni,  32, 8, 2, 4, drop=drop),
        #     SepConv1d(    32,  64, 8, 2, 4, drop=drop),
        #     SepConv1d(    64, 128, 8, 4, 4, drop=drop),
        #     SepConv1d(   128, 128, 8, 4, 4, drop=drop),
        #     SepConv1d(   128, 256, 8, 2, 3),
        #     Flatten(),
        #     nn.Dropout(drop), nn.Linear(256, 64), nn.ReLU(inplace=True),
        #     nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            # nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        # fft_out = self.fft(t_fft)
        # t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(raw_out)
        return out    
#%% ==================   1dconv - 5 conv - 1 pool - 2 FC
class Classifier_1d_5_conv_1_pool_2_fc(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
        super().__init__()
        
        # assert int(n_flt) == n_flt
        flat_in = 512 * int (raw_size / (2*4*4*4*4*2))
        # assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
        # flat_in = 256*int(n_flt)
        
        
        if batch_norm:
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                # SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
                nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))    
                # nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))

        else:        
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop),
                # SepConv1d(   512,1024, 8, 4, 2),
                nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 128), nn.ReLU(inplace=True))              
                # nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out    
#%% ==================   1dconv - 5 conv - 1 pool - 2 FC - 1 output
class Classifier_1d_5_conv_1_pool_2_fc_1_out(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
        super().__init__()
        
        # assert int(n_flt) == n_flt
        flat_in = 512 * int (raw_size / (2*4*4*4*4*2))
        # assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
        # flat_in = 256*int(n_flt)
        
        
        if batch_norm:
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                # SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
                nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))    
                # nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))

        else:        
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 2, 3, drop=drop),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop),
                # SepConv1d(   512,1024, 8, 4, 2),
                nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 128), nn.ReLU(inplace=True))              
                # nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out
#%% ==================   1dconv - 5 conv - 1 pool - 2 FC - kernel1 4
class Classifier_1d_5_conv_1_pool_2_fc_k1_4(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
        super().__init__()
        
        # assert int(n_flt) == n_flt
        flat_in = 512 * int (raw_size / (2*4*4*4*4*2))
        # assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
        # flat_in = 256*int(n_flt)
        
        
        if batch_norm:
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 4, 2, 1, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                # SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
                nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))    
                # nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))

        else:        
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 4, 2, 1, drop=drop),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop),
                # SepConv1d(   512,1024, 8, 4, 2),
                nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 128), nn.ReLU(inplace=True))              
                # nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out  
#%% ==================   1dconv - 5 conv - 1 pool - 2 FC - kernels 4
class Classifier_1d_5_conv_1_pool_2_fc_k_4(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
        super().__init__()
        
        # assert int(n_flt) == n_flt
        flat_in = 512 * int (raw_size / (2*4*4*4*4*2))
        # assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
        # flat_in = 256*int(n_flt)
        
        
        if batch_norm:
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 4, 2, 1, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
                SepConv1d(    32,  64, 4, 4, 0, drop=drop, batch_norm = batch_norm),
                SepConv1d(    64, 128, 4, 4, 0, drop=drop, batch_norm = batch_norm),
                SepConv1d(   128, 256, 4, 4, 0, drop=drop, batch_norm = batch_norm),
                SepConv1d(   256, 512, 4, 4, 0, drop=drop, batch_norm = batch_norm),
                # SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
                nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))    
                # nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))

        else:        
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 4, 2, 1, drop=drop),  #out: raw_size/str
                SepConv1d(    32,  64, 4, 4, 0, drop=drop),
                SepConv1d(    64, 128, 4, 4, 0, drop=drop),
                SepConv1d(   128, 256, 4, 4, 0, drop=drop),
                SepConv1d(   256, 512, 4, 4, 0, drop=drop),
                # SepConv1d(   512,1024, 8, 4, 2),
                nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 128), nn.ReLU(inplace=True))              
                # nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out  
#%% ==================   1dconv - 5 conv - 2 FC - strides 4
class Classifier_1d_5_conv_2_fc_str_4(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
        super().__init__()
        
        # assert int(n_flt) == n_flt
        flat_in = 512 * int (raw_size / (4*4*4*4*4))
        # assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
        # flat_in = 256*int(n_flt)
        
        
        if batch_norm:
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 4, 2, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                # SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
                # nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))    
                # nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))

        else:        
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 4, 2, drop=drop),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop),
                # SepConv1d(   512,1024, 8, 4, 2),
                # nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 128), nn.ReLU(inplace=True))              
                # nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out    
#%% ==================   1dconv - 5 conv - 2 FC - strides 4 - Sigmoid - 2 out
class Classifier_1d_5_conv_2_fc_str_4_sigm_2out(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
        super().__init__()
        
        # assert int(n_flt) == n_flt
        flat_in = 512 * int (raw_size / (4*4*4*4*4))
        # assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
        # flat_in = 256*int(n_flt)
        
        
        if batch_norm:
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 4, 2, drop=drop, batch_norm = batch_norm, activ=nn.Sigmoid),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop, batch_norm = batch_norm, activ=nn.Sigmoid),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop, batch_norm = batch_norm, activ=nn.Sigmoid),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop, batch_norm = batch_norm, activ=nn.Sigmoid),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm, activ=nn.Sigmoid),
                # SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
                # nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.Sigmoid())    
                # nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))

        else:        
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 4, 2, drop=drop, activ=nn.Sigmoid),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop, activ=nn.Sigmoid),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop, activ=nn.Sigmoid),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop, activ=nn.Sigmoid),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop, activ=nn.Sigmoid),
                # SepConv1d(   512,1024, 8, 4, 2),
                # nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 128), nn.Sigmoid())              
                # nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out    
#%% ==================   1dconv - 5 conv - 2 FC - strides 4 - Sigmoid - 1 out
class Classifier_1d_5_conv_2_fc_str_4_sigm_1out(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
        super().__init__()
        
        # assert int(n_flt) == n_flt
        flat_in = 512 * int (raw_size / (4*4*4*4*4))
        # assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
        # flat_in = 256*int(n_flt)
        
        
        if batch_norm:
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 4, 2, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                # SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
                # nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.Sigmoid())    
                # nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))

        else:        
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 4, 2, drop=drop),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop),
                # SepConv1d(   512,1024, 8, 4, 2),
                # nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 128), nn.Sigmoid())              
                # nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, 1))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out    
#%% ==================   1dconv - 5 conv - 2 FC - strides 4 - 1 out
class Classifier_1d_5_conv_2_fc_str_4_1out(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
        super().__init__()
        
        # assert int(n_flt) == n_flt
        flat_in = 512 * int (raw_size / (4*4*4*4*4))
        # assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
        # flat_in = 256*int(n_flt)
        
        
        if batch_norm:
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 4, 2, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                # SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
                # nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))    
                # nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))

        else:        
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 4, 2, drop=drop),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop),
                SepConv1d(   256, 512, 8, 4, 2, drop=drop),
                # SepConv1d(   512,1024, 8, 4, 2),
                # nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 128), nn.ReLU(inplace=True))              
                # nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, 1))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out    
#%% ==================   1dconv - 4 conv - 1 pool - 2 FC - strides 4
class Classifier_1d_4_conv_1_pool_2_fc_str_4(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
        super().__init__()
        
        # assert int(n_flt) == n_flt
        flat_in = 256 * int(raw_size / (4*4*4*4*2))
        # assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
        # flat_in = 256*int(n_flt)
        
        
        if batch_norm:
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 4, 2, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                # SepConv1d(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                # SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
                nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))    
                # nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))

        else:        
            self.raw = nn.Sequential(
                SepConv1d(raw_ni,  32, 8, 4, 2, drop=drop),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop),
                # SepConv1d(   256, 512, 8, 4, 2, drop=drop),
                # SepConv1d(   512,1024, 8, 4, 2),
                nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 128), nn.ReLU(inplace=True))              
                # nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out    
#%% ==================   1dconv - 4 conv - 2 FC - strides 4 - subsampled
class Classifier_1d_4_conv_2_fc_str_4_sub(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
        super().__init__()
        
        # assert int(n_flt) == n_flt
        flat_in = 256 * int(raw_size / (2*4*4*4*4))
        # assert int (raw_size / (2*4**3)) == (raw_size / (2*4**3))
        # flat_in = 256*int(n_flt)
        
        
        if batch_norm:
            self.raw = nn.Sequential(
                nn.MaxPool1d(1, 2), # Subsampling
                SepConv1d(raw_ni,  32, 8, 4, 2, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                # SepConv1d(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm),
                # SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
                # nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))    
                # nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))

        else:        
            self.raw = nn.Sequential(
                nn.MaxPool1d(1, 2), # Subsampling
                SepConv1d(raw_ni,  32, 8, 4, 2, drop=drop),  #out: raw_size/str
                SepConv1d(    32,  64, 8, 4, 2, drop=drop),
                SepConv1d(    64, 128, 8, 4, 2, drop=drop),
                SepConv1d(   128, 256, 8, 4, 2, drop=drop),
                # SepConv1d(   256, 512, 8, 4, 2, drop=drop),
                # SepConv1d(   512,1024, 8, 4, 2),
                # nn.MaxPool1d(2, 2),
                Flatten(),
                nn.Dropout(drop), nn.Linear(flat_in, 128), nn.ReLU(inplace=True))              
                # nn.Dropout(drop), nn.Linear( 128, 128), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out    
#%% ==================   1dconv - 4 conv - 2 FC - strides 4 - kernels 4 - subsampled
class Classifier_1d_4_conv_2_fc_str_4_k_4_sub(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True):
        super().__init__()
        
        flat_in = 256 * int(raw_size / (2*4*4*4*4))

        self.raw = nn.Sequential(
            nn.MaxPool1d(1, 2), # Subsampling
            SepConv1d(raw_ni,  32, 4, 4, 0, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
            SepConv1d(    32,  64, 4, 4, 0, drop=drop, batch_norm = batch_norm),
            SepConv1d(    64, 128, 4, 4, 0, drop=drop, batch_norm = batch_norm),
            SepConv1d(   128, 256, 4, 4, 0, drop=drop, batch_norm = batch_norm),
            # SepConv1d(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm),
            # SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
            # nn.MaxPool1d(2, 2),
            Flatten(),
            nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))    
            # nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out
#%% ==================   1dconv - 4 conv - 2 FC - strides 4 - channel depth halved - subsampled
class Classifier_1d_4_conv_2_fc_str_4_half_chan_sub(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True):
        super().__init__()
        
        flat_in = 128 * int(raw_size / (2*4*4*4*4))
        
        self.raw = nn.Sequential(
            nn.MaxPool1d(1, 2), # Subsampling
            SepConv1d(raw_ni,  16, 8, 4, 2, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
            SepConv1d(    16,  32, 8, 4, 2, drop=drop, batch_norm = batch_norm),
            SepConv1d(    32,  64, 8, 4, 2, drop=drop, batch_norm = batch_norm),
            SepConv1d(    64, 128, 8, 4, 2, drop=drop, batch_norm = batch_norm),
            # SepConv1d(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm),
            # SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
            # nn.MaxPool1d(2, 2),
            Flatten(),
            nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))    
            # nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out
#%% ==================   1dconv - 4 conv - 2 FC - kernels [4,8,8,8] - strides [2,4,4,4] - pool [2,0,0,0] - channel depth halved - subsampled
class Classifier_4c_2f_s1_2_k1_4_p1_2_half_chan_sub(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True):
        super().__init__()
        
        flat_in = 128 * int(raw_size / (2*2*2*4*4*4))

        self.raw = nn.Sequential(
            nn.MaxPool1d(1, 2), # Subsampling
            SepConv1d(raw_ni,  16, 4, 2, 1, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
            nn.MaxPool1d(2, 2),
            SepConv1d(    16,  32, 8, 4, 2, drop=drop, batch_norm = batch_norm),
            SepConv1d(    32,  64, 8, 4, 2, drop=drop, batch_norm = batch_norm),
            SepConv1d(    64, 128, 8, 4, 2, drop=drop, batch_norm = batch_norm),
            # SepConv1d(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm),
            # SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
            # nn.MaxPool1d(2, 2),
            Flatten(),
            nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))    
            # nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out
#%% ==================   1dconv - 4 conv - 2 FC - kernels [16,8,8,8] - strides [8,4,4,4] - subsampled
class Classifier_4c_2f_k1_16_s1_8_sub(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True):
        super().__init__()
        
        flat_in = 256 * int(raw_size / (2*8*4*4*4))

        self.raw = nn.Sequential(
            nn.MaxPool1d(1, 2), # Subsampling
            SepConv1d(raw_ni,  32, 16, 8, 4, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
            SepConv1d(    32,  64,  8, 4, 2, drop=drop, batch_norm = batch_norm),
            SepConv1d(    64, 128,  8, 4, 2, drop=drop, batch_norm = batch_norm),
            SepConv1d(   128, 256,  8, 4, 2, drop=drop, batch_norm = batch_norm),
            # SepConv1d(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm),
            # SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
            # nn.MaxPool1d(2, 2),
            Flatten(),
            nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))    
            # nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out
#%% ==================   1dconv - 4 conv - 2 FC - kernels [8,8,8,4] - strides [4,4,4,2] - subsampled
class Classifier_4c_2f_k4_4_s4_2_sub(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True):
        super().__init__()
        flat_in = 256 * int(raw_size / (2*4*4*4*2))
        
        self.raw = nn.Sequential(
            nn.MaxPool1d(1, 2), # Subsampling
            SepConv1d(raw_ni,  32, 8, 4, 2, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
            SepConv1d(    32,  64, 8, 4, 2, drop=drop, batch_norm = batch_norm),
            SepConv1d(    64, 128, 8, 4, 2, drop=drop, batch_norm = batch_norm),
            SepConv1d(   128, 256, 4, 2, 1, drop=drop, batch_norm = batch_norm),
            # SepConv1d(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm),
            # SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
            # nn.MaxPool1d(2, 2),
            Flatten(),
            nn.Linear(flat_in, 128), nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))    
            # nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(128, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out    
#%% ==================   1dconv - 2 conv - 2 FC - strides 4 - subsampled 4
class Classifier_1d_2_conv_2_fc_sub4(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True):
        super().__init__()
        
        flat_in = 16 * int(raw_size / (4*4*4))
        
        self.raw = nn.Sequential(
            nn.MaxPool2d(1, 4), # Subsampling
            SepConv1d_v4(raw_ni,  16, 8, 4, 2, drop=0.2, batch_norm = batch_norm),  #out: raw_size/str
            SepConv1d_v4(     16, 16, 8, 4, 2, drop=0.2, batch_norm = batch_norm),
            Flatten(),
            nn.Linear(flat_in, 16), nn.Dropout(drop), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(16, no))
        
    def forward(self, t_raw):
        t_raw  =  t_raw.unsqueeze(2)
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out    
#%% ==================   felexile net          R2Q: ready to quantize: drop after relu + BN2d
class Classifier_1d_flex_net(nn.Module):
    def __init__(self, raw_ni, no, raw_size, net,drop=.5 ,
                 batch_norm = True, conv_type = '2d'):
        
        super().__init__()
        
#        FC_neu = 64
        pre = net['pre'] if 'pre' in net else None
        convs = net['conv']
        bias = net['bias'] if 'bias' in net else True
        FCs = net['fc']
        kernels = net['kernels']
        kernels = [kernels]*len(convs) if type(kernels)==int else kernels
        
        strides = net['strides']
        strides = [strides]*len(convs) if type(strides)==int else strides
        
        pads = net['pads']
        pads    =    [pads]*len(convs) if type(pads)==int else pads
        drop0 = 0.2

        flat_in = convs[len(convs)-1] * int(raw_size / np.prod(strides))
        
        params = zip(convs[:len(convs)-1], convs[1:], kernels, strides, pads)
#        if batch_norm:
        if pre:
            raw_layers = [nn.MaxPool2d(1, pre)] # Subsampling
            flat_in = flat_in // pre
        else:
            raw_layers = [] 
            
        raw_layers.append(SepConv1d_v6(raw_ni,  convs[0], kernels[0], strides[0], pads[0], 
                                  drop0, batch_norm, conv_type, bias = bias))  #out: raw_size/str
        [raw_layers.append(SepConv1d_v6(i_ch,  conv, kernel, strd, pad, drop0, 
                                        batch_norm, conv_type, bias = bias))\
                                        for  i_ch, conv, kernel, strd, pad in params]
        
        self.raw = nn.Sequential(*raw_layers)
                
#        FC_layers = Flatten()        
        
        self.FC = nn.Sequential(
            Flatten2(),
            nn.Linear(flat_in, FCs[0], bias = bias), nn.ReLU(inplace=True), nn.Dropout(drop),
#                nn.Linear( 128, 128),    nn.BatchNorm1d(num_features = 128), nn.Dropout(drop), nn.ReLU(inplace=True)
            )            
        
        self.out = nn.Sequential(
#            nn.Linear(128, 64), nn.ReLU(inplace=True), 
            nn.Linear(FCs[len(FCs)-1], no, bias = bias ))

        self.quant = QuantStub()
        
        self.dequant = DeQuantStub()
    
    def fuse_model(self):
        for m in self.modules():
            if type(m) == SepConv1d_v6:
                fuse_profile = ['layers.1', 'layers.2', 'layers.3']
#                fuse_profile = ['layers.0.pointwise', 'layers.1', 'layers.2']
                torch.quantization.fuse_modules(m, fuse_profile, inplace=True)
#                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
        torch.quantization.fuse_modules(self.FC, ['1','2'], inplace=True)
       
    def forward(self, t_raw):        
        t_raw = self.quant(t_raw)        
        t_raw  =  t_raw.unsqueeze(2)
        raw_out = self.raw(t_raw)
        FC_out = self.FC(raw_out)        
        out = self.out(FC_out)
        out  = self.dequant(out)
        return out   
#%% ==================   1-FC-layer simple NN
class Classifier_1f_sub4(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True):
        super().__init__()
        
        flat_in = raw_size // 4 * 2
        
        self.raw = nn.Sequential(
            Flatten())
            # nn.Linear(flat_in, 16), nn.Dropout(drop), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(flat_in, no))
        
    def forward(self, t_raw):
        t_raw  =  t_raw.unsqueeze(2)
        t_raw = nn.MaxPool2d(1, 4)(t_raw)
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out
#%% ==================   2-FC-layer simple NN
class Classifier_2f_sub4(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True):
        super().__init__()
        
        flat_in = raw_size // 4 * 2
        
        self.raw = nn.Sequential(
            Flatten(),
            nn.Dropout(0.2),
            nn.Linear(flat_in, 8), nn.Dropout(0.33), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(8, 1))
        
    def forward(self, t_raw):
        t_raw  =  t_raw.unsqueeze(2)
        t_raw = nn.MaxPool2d(1, 4)(t_raw)
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out
#%% ==================   2-FC-layer simple NN
class Classifier_2f_sub8(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True):
        super().__init__()
        
        flat_in = raw_size // 8 * 2
        
        self.raw = nn.Sequential(
            Flatten(),
            nn.Dropout(0.2),
            nn.Linear(flat_in, 8), nn.Dropout(0.33), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(8, 1))
        
    def forward(self, t_raw):
        t_raw  =  t_raw.unsqueeze(2)
        t_raw = nn.MaxPool2d(1, 8)(t_raw)
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out
#%% ==================   2-FC-layer simple NN
class Classifier_2f_sub16(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True):
        super().__init__()
        
        flat_in = raw_size // 16 * 2
        
        self.raw = nn.Sequential(
            Flatten(),
            nn.Dropout(0.2),
            nn.Linear(flat_in, 8), nn.Dropout(0.33), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(8, 1))
        
    def forward(self, t_raw):
        t_raw  =  t_raw.unsqueeze(2)
        t_raw = nn.MaxPool2d(1, 16)(t_raw)
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out
#%% ==================   2-FC-layer simple NN
class Classifier_2f_sub32(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True):
        super().__init__()
        
        flat_in = raw_size // 32 * 2
        
        self.raw = nn.Sequential(
            Flatten(),
            nn.Dropout(0.2),
            nn.Linear(flat_in, 8), nn.Dropout(0.33), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(8, 1))
        
    def forward(self, t_raw):
        t_raw  =  t_raw.unsqueeze(2)
        t_raw = nn.MaxPool2d(1, 32)(t_raw)
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out
#%% ==================   1 non-overlapping conv, 2 FC
class Classifier_1c_2f_sub4(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True):
        super().__init__()
        
        flat_in = raw_size // (4*4)
        
        self.raw = nn.Sequential(
            nn.MaxPool2d(1, 4), # Subsampling
            SepConv1d_v5(raw_ni, 1, 4, 4, 0, drop=0.2, batch_norm=True),
            Flatten(),
            nn.Linear(flat_in, 8), nn.Dropout(drop), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(8, no))
        
    def forward(self, t_raw):
        t_raw  =  t_raw.unsqueeze(2)
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out
#%% ==================   1 non-overlapping conv, 1 FC
class Classifier_1c_1f_sub4(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True):
        super().__init__()
        
        flat_in = raw_size // (4*4)
        
        self.raw = nn.Sequential(
            SepConv1d_v5(raw_ni, 1, 4, 4, 0, drop=0.2, batch_norm=True),
            Flatten())
            # nn.Linear(flat_in, 8), nn.Dropout(drop), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(flat_in, no))
        
    def forward(self, t_raw):
        t_raw  =  t_raw.unsqueeze(2)
        t_raw = nn.MaxPool2d(1, 4)(t_raw)
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out
#%% ==================   1 non-overlapping conv, 1 FC
class Classifier_1c_1f_sub8(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True):
        super().__init__()
        
        flat_in = raw_size // (8*4)
        
        self.raw = nn.Sequential(
            SepConv1d_v5(raw_ni, 1, 4, 4, 0, drop=0.2, batch_norm=True),
            Flatten())
            # nn.Linear(flat_in, 8), nn.Dropout(drop), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(flat_in, no))
        
    def forward(self, t_raw):
        t_raw  =  t_raw.unsqueeze(2)
        t_raw = nn.MaxPool2d(1, 8)(t_raw)
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out
#%% ==================   1 non-overlapping conv, 1 FC
class Classifier_1c_1f_sub8_1out(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True):
        super().__init__()
        
        flat_in = raw_size // (8*4)
        
        self.raw = nn.Sequential(
            SepConv1d_v5(raw_ni, 1, 4, 4, 0, drop=0.2, batch_norm=True),
            Flatten())
            # nn.Linear(flat_in, 8), nn.Dropout(drop), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(flat_in, 1))
        
    def forward(self, t_raw):
        t_raw  =  t_raw.unsqueeze(2)
        t_raw = nn.MaxPool2d(1, 8)(t_raw)
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out
#%% ==================   1 non-overlapping conv, 1 FC
class Classifier_1c_1f_sub16_1out(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True):
        super().__init__()
        
        flat_in = raw_size // (16*2)
        
        self.raw = nn.Sequential(
            SepConv1d_v5(raw_ni, 1, 2, 2, 0, drop=0.2, batch_norm=True),
            Flatten())
            # nn.Linear(flat_in, 8), nn.Dropout(drop), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(flat_in, 1))
        
    def forward(self, t_raw):
        t_raw  =  t_raw.unsqueeze(2)
        t_raw = nn.MaxPool2d(1, 16)(t_raw)
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out
#%% ==================   1 non-overlapping conv, 1 FC
class Classifier_1c_1f_sub16_1out_k4(nn.Module):
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = True):
        super().__init__()
        
        flat_in = raw_size // (16*4)
        
        self.raw = nn.Sequential(
            SepConv1d_v5(raw_ni, 1, 4, 4, 0, drop=0.2, batch_norm=True),
            Flatten())
            # nn.Linear(flat_in, 8), nn.Dropout(drop), nn.ReLU(inplace=True))
        
        self.out = nn.Sequential(
            nn.Linear(flat_in, 1))
        
    def forward(self, t_raw):
        t_raw  =  t_raw.unsqueeze(2)
        t_raw = nn.MaxPool2d(1, 16)(t_raw)
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out
class Classifier_ims_nn(nn.Module):
    def __init__(self):
        super().__init__()

        self.out = nn.Sequential(
            nn.Linear(13, 27),
            nn.Softsign(),
            nn.Linear(27, 2)
        )

    def forward(self, x):
        return self.out(x)