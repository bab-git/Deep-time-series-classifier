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
#from prune import *
#import argparse
from operator import itemgetter
from heapq import nsmallest
#import time



#=====================
class FilterPrunner:
    def __init__(self, model):
        self.model = model
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
        for module in self.model.raw.modules():            
            if type(module) == nn.Conv1d or type(module) == nn.Conv2d:
                x = module(x)
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1
        return self.model.FC(x.view(x.size(0), -1))
    
    
    def compute_rank(self, grad):
    	activation_index = len(self.activations) - self.grad_index - 1
    	activation = self.activations[activation_index]
    	values = \
    		torch.sum((activation * grad), dim = 0).\
    			sum(dim=2).sum(dim=3)[0, :, 0, 0].data
    	
    	# Normalize the rank by the filter dimensions
    	values = \
    		values / (activation.size(0) * activation.size(2) * activation.size(3))
    
    	if activation_index not in self.filter_ranks:
    		self.filter_ranks[activation_index] = \
    			torch.FloatTensor(activation.size(1)).zero_().cuda()
    
    	self.filter_ranks[activation_index] += values
    	self.grad_index += 1
        
        



#---------------------------------        
class PrunningFineTuner:
    def __init__(self, trn_dl, val_dl, model):
        self.train_data_loader = trn_dl
        self.test_data_loader = val_dl
        
        self.model = model
        self.criterion = nn.CrossEntropyLoss (reduction = 'sum')
        self.prunner = FilterPrunner(self.model) 
        self.model.train()
        
    def test(self):
#        return         
        self.model.eval()
        correct = 0
        total = 0
        
        for i, (batch, label) in enumerate(self.test_data_loader):
#            if args.use_cuda:
#                batch = batch.cuda()
            output = self.model(batch)                        
            pred  = output.argmax(dim=1)
#            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label.cpu()).sum()
            total += label.size(0)
        
        print("Accuracy :", float(correct) / total)
        
        self.model.train()
    
    
    def prune(self):
        #Get the accuracy before prunning
        self.test()
    