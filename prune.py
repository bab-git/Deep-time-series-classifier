#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 18:20:02 2020

@author: bhossein
"""

import torch
#from torch.autograd import Variable
#from torchvision import models
#import cv2
#import sys
import numpy as np

#%%=====================
def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]

def trim_module(new_module, module, filter_index, module_type):
    
    old_weights = module.weight.data.cpu().numpy()  
    new_weights = new_module.weight.data.cpu().numpy()
    
    if module_type == "conv_out":

        new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
        new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
        
    elif module_type == "conv_in":

        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]        
            
    elif module_type == "BN":
        new_module = \
            torch.nn.BatchNorm2d(num_features = module.num_features-1)
            
        rm_numpy = module.running_mean.data.cpu().numpy()    
        rm = np.zeros(shape = (rm_numpy.shape[0] - 1), dtype = np.float32)
        rm[:filter_index] = rm_numpy[:filter_index]
        rm[filter_index : ] = rm_numpy[filter_index + 1 :]
        new_module.running_mean.data = torch.from_numpy(rm)

    elif module_type == "lin_out":
        new_weights[: filter_index, :] = old_weights[: filter_index, :]
        new_weights[filter_index : , :] = old_weights[filter_index + 1 :, :]
    
    new_module.weight.data = torch.from_numpy(new_weights)
        
    if module_type != "conv_in":
        bias_numpy = module.bias.data.cpu().numpy()    
        bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
        bias[:filter_index] = bias_numpy[:filter_index]
        bias[filter_index : ] = bias_numpy[filter_index + 1 :]
        new_module.bias.data = torch.from_numpy(bias)
           
    if torch.cuda.is_available():
        new_module.cuda()
        
        
    return new_module
#        new_conv.weight.data = new_conv.weight.data.cuda()
#    if torch.cuda.is_available():
#        new_conv.bias.data = new_conv.bias.data.cuda()
    
    
    

def prune_conv_layers(model, layer_index, filter_index):
#    _, conv = list(model.raw._modules.items())[layer_index]
    if torch.cuda.is_available():
        model = model.cuda()
    
    model_l = len(model.raw)
    
    if type(model.raw[0]) == torch.nn.MaxPool2d:    
        layer = 0
        modules = model.raw[0]
        layer_range = range(1,model_l)
    else:
        layer = -1
        modules = []
        layer_range = range(0,model_l)
        
    for i_layer in layer_range:              #net.{raw,FC,out}
        for (name,module) in model.raw[i_layer].layers._modules.items():
            layer += 1
            modules = np.append(modules, module)
    
    conv = modules[layer_index]
    next_module =  modules[layer_index+1]
    
    BN = None
    if isinstance(next_module, torch.nn.modules.batchnorm.BatchNorm2d):
        BN = next_module
        
    next_conv = None
    offset = 1
    
    while layer_index + offset <  len(modules):
        res =  modules[layer_index+offset]
        if isinstance(res, torch.nn.modules.conv.Conv2d):
            next_conv = res
            next_conv_width = modules[layer_index+offset+1]
            break        
        offset += 1        
    
    if not next_conv is None and offset == 1: 
        # first depthwise convolution
        raise BaseException("No linear layer found in classifier")
        
        
    
    # layer_Conv : assmed to be the widthwise conv.-layer                        
    new_conv= \
        torch.nn.Conv2d(in_channels = conv.in_channels, \
            out_channels = conv.out_channels - 1,
            kernel_size = conv.kernel_size, \
            stride = conv.stride,
            padding = conv.padding,
            dilation = conv.dilation,
            groups = conv.groups,
            bias = (conv.bias is not None))    
    
    new_conv = trim_module(new_conv, conv, filter_index, "conv_out")        
#    old_weights = conv.weight.data.cpu().numpy()
#    new_weights = new_conv.weight.data.cpu().numpy()
    
#    new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
#    new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
#    new_conv.weight.data = torch.from_numpy(new_weights)
        
#    if torch.cuda.is_available():
#        new_conv.weight.data = new_conv.weight.data.cuda()
        
#    bias_numpy = conv.bias.data.cpu().numpy()
    
#    bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
#    bias[:filter_index] = bias_numpy[:filter_index]
#    bias[filter_index : ] = bias_numpy[filter_index + 1 :]
#    new_conv.bias.data = torch.from_numpy(bias)
#    if torch.cuda.is_available():
#        new_conv.bias.data = new_conv.bias.data.cuda()
        
     
            
    # BatchNr after depthwise convolution
    if not BN is None:
        new_BN = \
            torch.nn.BatchNorm2d(num_features = BN.num_features-1)
        
        new_BN = trim_module(new_BN, BN, filter_index, "BN")
#        
#        old_weights = BN.weight.data.cpu().numpy()
#        new_weights = new_BN.weight.data.cpu().numpy()    
#        new_weights[: filter_index] = old_weights[: filter_index]
#        new_weights[filter_index :] = old_weights[filter_index + 1 :]        
#        new_BN.weight.data = torch.from_numpy(new_weights)        
#        if torch.cuda.is_available():
#            new_BN.weight.data = new_BN.weight.data.cuda()
#            
#        bias_numpy = BN.bias.data.cpu().numpy()    
#        bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
#        bias[:filter_index] = bias_numpy[:filter_index]
#        bias[filter_index : ] = bias_numpy[filter_index + 1 :]
#        new_BN.bias.data = torch.from_numpy(bias)
#        if torch.cuda.is_available():
#            new_BN.bias.data = new_BN.bias.data.cuda()
        
    # Following conv.-layer
    if not next_conv is None:
        # assmed to be the depthwise conv.-layer
        next_new_conv = \
            torch.nn.Conv2d(in_channels = next_conv.in_channels - 1,\
                out_channels =  next_conv.out_channels-1, \
                kernel_size = next_conv.kernel_size, \
                stride = next_conv.stride,
                padding = next_conv.padding,
                dilation = next_conv.dilation, 
                groups = next_conv.groups-1,
                bias = (next_conv.bias is not None))
            
        next_new_conv = trim_module(next_new_conv, next_conv, filter_index, "conv_out")
            
#        old_weights = next_conv.weight.data.cpu().numpy()
#        new_weights = next_new_conv.weight.data.cpu().numpy()
#        
#        new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
#        new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
#        next_new_conv.weight.data = torch.from_numpy(new_weights)
#        if torch.cuda.is_available():
#            next_new_conv.weight.data = next_new_conv.weight.data.cuda()
#        
        
#        bias_numpy = next_conv.bias.data.cpu().numpy()
#        bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
#        bias[:filter_index] = bias_numpy[:filter_index]
#        bias[filter_index : ] = bias_numpy[filter_index + 1 :]
#    
#        next_new_conv.bias.data = torch.from_numpy(bias)
#        if torch.cuda.is_available():
#            next_new_conv.bias.data = next_new_conv.bias.data.cuda()        
                
        # assmed to be the next widthwise conv.-layer
        next_new_conv_width = \
            torch.nn.Conv2d(in_channels = next_conv_width.in_channels - 1,\
                out_channels =  next_conv_width.out_channels, \
                kernel_size = next_conv_width.kernel_size, \
                stride = next_conv_width.stride,
                padding = next_conv_width.padding,
                dilation = next_conv_width.dilation,                
#                groups = next_conv.groups,
                groups = next_conv_width.groups,
                bias = (next_conv_width.bias is not None))
        next_new_conv_width = trim_module(next_new_conv_width, next_conv_width, filter_index, "conv_in")
            
#        old_weights = next_conv_width.weight.data.cpu().numpy()
#        new_weights = next_new_conv_width.weight.data.cpu().numpy()
#        
#        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
#        new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]        
#        next_new_conv_width.weight.data = torch.from_numpy(new_weights)
#        if torch.cuda.is_available():
#            next_new_conv_width.weight.data = next_new_conv_width.weight.data.cuda()
#        
#        next_new_conv_width.bias.data = next_conv_width.bias.data
                                      
    
    # Updating the model        
    i_raw = np.floor((layer_index-1)/5)+1
    i_raw_layer = layer_index-(np.floor((layer_index-1)/5)*5+1)
    model.raw[int(i_raw)].layers[int(i_raw_layer)] = new_conv
    model.raw[int(i_raw)].layers[int(i_raw_layer)+1] = new_BN
    if not next_conv is None:
                        
        model.raw[int(i_raw)+1].layers[0] = next_new_conv
        model.raw[int(i_raw)+1].layers[1] = next_new_conv_width
                
#        del model.features
        del conv, next_conv, next_conv_width
        
#        model.features = features
    else:
        #Prunning the last conv layer. This affects the first linear layer of the classifier.
        layer_index = 0
        old_linear_layer = None
        for _, module in model.FC._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index = layer_index  + 1
            
        if old_linear_layer is None:
            raise BaseException("No linear layer found in classifier")
        
        params_per_input_channel = old_linear_layer.in_features // conv.out_channels
        
        new_linear_layer = \
            torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel, 
                old_linear_layer.out_features)
        
        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()        

        new_weights[:, : filter_index * params_per_input_channel] = \
            old_weights[:, : filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel :] = \
            old_weights[:, (filter_index + 1) * params_per_input_channel :]
        
        new_linear_layer.bias.data = old_linear_layer.bias.data

        new_linear_layer.weight.data = torch.from_numpy(new_weights)
        if torch.cuda.is_available():
            new_linear_layer.weight.data = new_linear_layer.weight.data.cuda()
            
        model.FC[1] = new_linear_layer
        
        del conv
            
    return model


#%%============================= prune_lin_layers ======================================================

def prune_lin_layers(model, layer_index, filter_index):
#    _, conv = list(model.raw._modules.items())[layer_index]
    if torch.cuda.is_available():
        model = model.cuda()
#    layer = 0

    modules = model.FC
#    _, modules = model.FC._modules.items()
#    for i_layer in range(1,5):              #4c2fc_sub2
#        for (name,module) in model.raw[i_layer].layers._modules.items():
#            layer += 1
#            modules = np.append(modules, module)
    
    lin = modules[1]
    next_lin = None
#    next_module =  modules[layer_index+1]
#    
#    BN = None
#    if isinstance(next_module, torch.nn.modules.batchnorm.BatchNorm2d):
#        BN = next_module
        
#    next_conv = None
#    offset = 1
#    
#    while layer_index + offset <  len(modules):
#        res =  modules[layer_index+offset]
#        if isinstance(res, torch.nn.modules.conv.Conv2d):
#            next_conv = res
#            next_conv_width = modules[layer_index+offset+1]
#            break        
#        offset += 1        
#    
#    if not next_conv is None and offset == 1: 
#        # first depthwise convolution
#        raise BaseException("No linear layer found in classifier")
        
        
    
    # layer_Conv : assmed to be the widthwise conv.-layer                        
    new_lin= \
        torch.nn.Linear(in_features = lin.in_features, 
                        out_features = lin.out_features-1,
                        bias = (lin.bias is not None))
    
    new_lin = trim_module(new_lin, lin, filter_index, "lin_out")
                         
    # BatchNr after depthwise convolution
#    if not BN is None:
#        new_BN = \
#            torch.nn.BatchNorm2d(num_features = BN.num_features-1)
#        
#        new_BN = trim_module(new_BN, BN, filter_index, "BN")

        
    # Following conv.-layer
#    if not next_conv is None:
#        # assmed to be the depthwise conv.-layer
#        next_new_conv = \
#            torch.nn.Conv2d(in_channels = next_conv.in_channels - 1,\
#                out_channels =  next_conv.out_channels-1, \
#                kernel_size = next_conv.kernel_size, \
#                stride = next_conv.stride,
#                padding = next_conv.padding,
#                dilation = next_conv.dilation, 
#                groups = next_conv.groups-1,
#                bias = (next_conv.bias is not None))
#            
#        next_new_conv = trim_module(next_new_conv, next_conv, filter_index, "conv_out")            
#                
#        # assmed to be the next widthwise conv.-layer
#        next_new_conv_width = \
#            torch.nn.Conv2d(in_channels = next_conv_width.in_channels - 1,\
#                out_channels =  next_conv_width.out_channels, \
#                kernel_size = next_conv_width.kernel_size, \
#                stride = next_conv_width.stride,
#                padding = next_conv_width.padding,
#                dilation = next_conv_width.dilation,                
##                groups = next_conv.groups,
#                groups = next_conv_width.groups,
#                bias = (next_conv_width.bias is not None))
#        next_new_conv_width = trim_module(next_new_conv_width, next_conv_width, filter_index, "conv_in")
            
    
    # Updating the model            
    model.FC[1] = new_lin
    if not next_lin is None:
                        
#        model.raw[int(i_raw)+1].layers[0] = next_new_conv
#        model.raw[int(i_raw)+1].layers[1] = next_new_conv_width
                
        del conv, next_conv, next_conv_width
        
    else:
        #Prunning the output linear layer
#        layer_index = 0
#        old_linear_layer = None
#        for _, module in model.FC._modules.items():
#            if isinstance(module, torch.nn.Linear):
#                old_linear_layer = module
#                break
#            layer_index = layer_index  + 1
        old_linear_layer = model.out[0]
        
        if old_linear_layer is None:
            raise BaseException("No linear layer found in classifier")
        
#        params_per_input_channel = old_linear_layer.in_features // conv.out_channels
        
        new_linear_layer = \
            torch.nn.Linear(old_linear_layer.in_features - 1, 
                old_linear_layer.out_features)
        
        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()        

        new_weights[:, : filter_index] = old_weights[:, : filter_index]
        new_weights[:, filter_index :] = old_weights[:, filter_index + 1:]
        
        new_linear_layer.bias.data = old_linear_layer.bias.data

        new_linear_layer.weight.data = torch.from_numpy(new_weights)
        if torch.cuda.is_available():
            new_linear_layer.weight.data = new_linear_layer.weight.data.cuda()
            
        model.out[0] = new_linear_layer
        
        del lin
            
    return model