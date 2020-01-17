#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:53:16 2019

@author: bhossein
"""

#import torch
from torch import nn

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
    
#%% ==================          SepConv1d
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
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None):
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
    def __init__(self, raw_ni, no, raw_size, drop=.5, batch_norm = None, conv_type = '1d'):
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

#%% ==================   1dconv - 3 conv - 2 FC
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
        raw_out = self.raw(t_raw)
#        fft_out = self.fft(t_fft)
#        t_in = torch.cat([raw_out, fft_out], dim=1)
        out = self.out(raw_out)
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

