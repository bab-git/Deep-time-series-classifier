#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:12:27 2020

@author: bhossein
"""

from torch import nn

import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import dataset
from prune import *
#import argparse
from operator import itemgetter
from heapq import nsmallest
#import time

import numpy as np

import pickle

#=====================
class FilterPrunner:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.reset()
        
    def reset(self):
        self.filter_ranks = {}
        
    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}
        
        activation_index = 0
#        for layer, (name, module) in enumerate (self.model.raw.modules()):
#        for layer, module in enumerate(self.model.raw.modules()):
        layer = 0
        x = x.unsqueeze(2)
        module = self.model.raw[0]
        x =module(x)
        for i_layer in range(1,5):              #4c2fc_sub2
            for (name,module) in self.model.raw[i_layer].layers._modules.items():
                layer += 1
                x = module(x)
#                modules = np.append(modules, module)
                if (type(module) == nn.Conv1d or type(module) == nn.Conv2d) and module.in_channels != module.out_channels:
                    x.register_hook(self.compute_rank)
                    self.activations.append(x)
                    self.activation_to_layer[activation_index] = layer
                    activation_index += 1
#        self.modules = modules
        return nn.Sequential(self.model.FC,self.model.out)(x.view(x.size(0), -1))
          
    
    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        taylor = activation * grad
        
        # Get the average value for every filter, 
        # accross all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data
                	
    	# Normalize the rank by the filter dimensions
        taylor = \
            taylor / (activation.size(0) * activation.size(2) * activation.size(3))
    
        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_()

            if torch.cuda.is_available():
#            if args.use_cuda:
#            self.filter_ranks[activation_index] = self.filter_ranks[activation_index].to(self.device)
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()
#                print(self.filter_ranks[activation_index].device)

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1
    
    
    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i]).cpu()
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cuda()
    

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
                
        return nsmallest(num, data, itemgetter(2))
                            
    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune            
        
        



# =========================== PrunningFineTuner ====================
class PrunningFineTuner:
    def __init__(self, trn_dl, val_dl, model,device = 'cpu' ):
        self.train_data_loader = trn_dl
        self.test_data_loader = val_dl
        
        self.device = device
        if torch.cuda.is_available():
            model = model.cuda()
        
        self.model = model        
        self.criterion = nn.CrossEntropyLoss (reduction = 'sum')
        self.prunner = FilterPrunner(self.model, self.device) 
        self.model.train()
        
    def test(self):
#        return        
        print(" Getting accuracy ")
        self.model.eval()
        correct = 0
        total = 0
        
        for i, (batch, label) in enumerate(self.test_data_loader):
#            if args.use_cuda:
            batch = batch.cuda()
            output = self.model(batch)                        
            pred  = output.argmax(dim=1)
#            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label.cpu()).sum()
            total += label.size(0)
        
        print("Accuracy :", float(correct) / total)
        
        self.model.train()
    
    
    def train(self, optimizer = None, epoches=10):
#        print("train_called")
        if optimizer is None:
#            optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.9)
            optimizer = optim.Adam(self.model.parameters(), lr = 0.001)

        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()
        print("Finished fine tuning.")

    
    def train_epoch(self, optimizer = None, rank_filters = False):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(optimizer, batch, label, rank_filters)

    def train_batch(self, optimizer, batch, label, rank_filters):

#        if args.use_cuda:
        if torch.cuda.is_available():
            batch = batch.cuda()
            label = label.cuda()
#            batch, label = [t.to(self.device) for t in (batch, label)]
        self.model.zero_grad()
        input = Variable(batch)

        if rank_filters:
            output = self.prunner.forward(input)
#            print(output.shape)
            self.criterion(output, Variable(label)).backward()
        else:
            self.criterion(self.model(input), Variable(label)).backward()
            optimizer.step()

    def total_num_filters(self):
        filters = 0
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.conv.Conv2d) or isinstance(module, torch.nn.modules.conv.Conv1d):
                filters = filters + module.out_channels
        return filters            
        
    
    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(rank_filters = True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def prune(self):
        #Get the accuracy before prunning
        self.test()
        self.model.train()
        
        #Make sure all the layers are trainable
        for param in self.model.parameters():
            param.requires_grad = True
            
        number_of_filters = self.total_num_filters()
        
        num_filters_to_prune_per_iteration = 32
        
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
        
        iterations = int(iterations * 2.0 / 3)
        
        print("Number of prunning iterations to reduce ??% filters: ", iterations)
                        
        for i_iter in range(iterations):
            print("Ranking filters...  iteration: ",i_iter)
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            
        
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1 
#
            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")
            model = self.model.cpu()

#            return prune_targets
            
            for layer_index, filter_index in prune_targets:
                model = prune_conv_layers(model, layer_index, filter_index)
#           
            
            self.model = model
            if torch.cuda.is_available():
                self.model = self.model.cuda()
#
            message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters prunned", str(message))
            
            self.test()
            
            print("Fine tuning to recover from prunning iteration.")
#            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
            self.train(optimizer, epoches = 10)    
            
            self.test()
            
            return self.model
    
        print("Finished. Going to fine tune the model a bit more")
        self.train(optimizer, epoches=15)
#        torch.save(model.state_dict(), "model_prunned")
        pickle.dump(model,open('model_prunned.pth','wb'))
    
    
# %% test
#for name, module in model._modules.items():
#filters = 0
#for module in model.modules():
#    print('=============================')
#    print(module)
#    if isinstance(module, torch.nn.modules.conv.Conv2d) or isinstance(module, torch.nn.modules.conv.Conv1d):
#        filters = filters + module.out_channels
#print(filters)
            
#for layer, (name, module) in enumerate(model_VGG16.features._modules.items()):
#i_batch, batch = next(enumerate(trn_dl))
#x_raw, y_target = batch
#x = x_raw
#for layer, module in enumerate(model.modules()):
#for i_layer in range(5):  #4c2fc
#    for 
#        
#model.out    
##    print ()
##    x = module(x)
##    if type(module) == nn.Conv1d or type(module) == nn.Conv2d:
#        print (layer,"  ", name,"  ", module)
#        print (" ")
#            