#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:12:40 2019

@author: bhossein
"""

#import time
#from collections import defaultdict
#from functools import partial
#from multiprocessing import cpu_count
#from pathlib import Path
#from textwrap import dedent
#import matplotlib.pyplot as plt
#import numpy as np
#import pandas as pd

#from sklearn.externals import joblib
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
#from torch import nn
#from torch import optim
#from torch.nn import functional as F
#from torch.optim.lr_scheduler import _LRScheduler
#from torch.utils.data import TensorDataset, DataLoader
from torch.utils import data

import wavio
# %%==============    

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'    
    def __init__(self,list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
#        self.path = path
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        
        # Load data and get label
        if index < 8000:
            main_path = '/vol/hinkelstn/data/FILTERED/atrial_fibrillation_8k/'
        else:
            main_path = '/vol/hinkelstn/data/FILTERED/sinus_rhythm_8k/'
            
#        list_f = os.listdir(main_path)        
        path = main_path+ID
        w = wavio.read(path)        
        X = w.data
        
        y = self.labels[index]
        
        return X, y