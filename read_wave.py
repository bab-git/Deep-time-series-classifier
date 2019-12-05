#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:28:34 2019

@author: bhossein
"""
import numpy as np
import wavio
import os
import matplotlib.pyplot as plt

from my_data_classes import create_datasets, create_loaders, read_data, create_datasets_file, smooth

#os.chdir('/home/bhossein/BMBF project/code_repo')

#%%==============================
plt.close('all')

n = 8
sub_dim = [2*2,4]

plt.figure(figsize=(18,10))
plt.subplots_adjust(wspace = 0.2, hspace = 0.5)

i_file = np.random.randint(8000, size = int(n/2))
i_file[0] =  4225 #Unwanted peeks
i_file[1] =  1869 #Unwanted peeks

AF_file_list = []
SIN_file_list = []

for i_data in range(n):
    if i_data < n/2:        
        main_path = 'C:\Hinkelstien/data/FILTERED/sinus_rhythm_8k/'
        plt_color = 'b'
        plt_title = 'Sinus'
        i_f = i_data
        i_plt_row = i_data+1
        i_plt_cl = 4+i_data+1
    else:
        main_path = 'C:\Hinkelstien/data/FILTERED/atrial_fibrillation_8k/'    
        plt_color = 'r'
        plt_title = 'AF'
        i_f = int(i_data-n/2)
        i_plt_row = 4+i_data+1
        i_plt_cl = 8+i_data+1

    list_f = os.listdir(main_path)
    file = list_f[i_file[i_f]]
    path = main_path+file        
      
    w = wavio.read(path)
    w.data = w.data[40000:50000,:]
  
    plt.subplot(4,4,i_plt_row)
    #plt.figure(figsize=(8,6))
    plt.plot(w.data[:,0],color = plt_color)
    plt.title(plt_title+', sample_id:'+str(i_file[i_f]))
    plt.subplot(4,4,i_plt_cl)
    plt.plot(w.data[:,1],color = plt_color)
    plt.xlabel('samples')
    
    
#    plt.figure()
#    plt.scatter(range(705),w.data[34745:35450,0],color = plt_color)
<<<<<<< HEAD
    
assert 1==2
=======
    if i_data <n/2:
        AF_file_list.append(file)
    else:
        SIN_file_list.append(file)
#AF_file_list = [{}]        
assert 1== 61090
>>>>>>> 11f589090fd7d9e05d9a51e20f20b5ecbc40af13

# %%================= individual files  
    
           
#i_file = 2
i_file = np.random.randint(8000, size = 1)

i_class = 1 #1:normal  2:atrial
#i_class = np.random.randint(2, size = 1)+1    
    
#path = '/data/BMBF/sample/filtered atrial waves/8aae6985-c0b9-41d4-ac89-9f721b8019d2.wav'
if i_class==1:
    main_path = 'C:\Hinkelstien/data/FILTERED/sinus_rhythm_8k/'
    plt_color = 'b'
    plt_title = 'normal sinus rhythm'
else:
    main_path = 'C:\Hinkelstien/data/FILTERED/atrial_fibrillation_8k/'    
    plt_color = 'r'
    plt_title = 'Atrial Fibrilation'
    
list_f = os.listdir(main_path)
file = list_f[i_file[0]]
path = main_path+file

#==============================
w = wavio.read(path)
w.data = w.data[30000:50000,:]

plt.figure(figsize=(8,6))
#plt.subplots(2,1,1)
plt.subplot(311)
#plt.figure(figsize=(8,6))
plt.plot(w.data[:,0],color = plt_color)
plt.title(plt_title+', sample_id:'+str(i_file))
plt.subplot(312)
plt.plot(w.data[:,1],color = plt_color)
plt.subplot(313)
plt.plot(w.data[:,1]-w.data[:,0],color = plt_color)
plt.xlabel('samples')

#plt.figure()
#plt.scatter(range(705),w.data[34745:35450,0],color = plt_color)

assert w.data.shape[0] == 61090
    
# %%================= heart py
import heartpy as hp
from scipy.signal import resample
#import matplotlib.pyplot as plt

sample_rate = 256
#sample_rate = 128

#data_org = hp.get_data('e0103.csv')
#data_org = hp.get_data('e0110.csv')
#data_org = data_org[10000:15000]
data_org = np.array(w.data[10000:,1])
#data = (data_org.copy()-data_org.min())/(data_org.max()-data_org.min())
#data = data_org-data_org.min()

data = data_org

plt.figure(figsize=(12,4))
plt.plot(data)

filtered = hp.filter_signal(data_org, cutoff = 0.05, sample_rate = sample_rate, filtertype='notch', order = 2)

#resample the data. Usually 2, 4, or 6 times is enough depending on original sampling rate
resampled_data = resample(filtered, len(filtered) * 2)

plt.figure(figsize=(12,4))
plt.plot(filtered)

plt.figure(figsize=(12,4))
plt.plot(resampled_data)

plt.figure(figsize=(12,4))
plt.plot(hp.scale_data(resampled_data))

#plt.figure(figsize=(12,4))
#plt.plot(data)


#wd, m = hp.process(data, sample_rate)
#wd, m = hp.process(hp.scale_data(filtered), sample_rate)
wd, m = hp.process(hp.scale_data(resampled_data), sample_rate * 2)

#visualise in plot of custom size
plt.figure(figsize=(12,4))
hp.plotter(wd, m)

#display computed measures
for measure in m.keys():
    print('%s: %f' %(measure, m[measure]))
