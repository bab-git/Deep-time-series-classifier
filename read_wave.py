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
import scipy

from my_data_classes import create_datasets, create_loaders, read_data, create_datasets_file, smooth, wave_harsh_peaks
import my_data_classes

os.chdir('/home/bhossein/BMBF project/code_repo')

#%%==============================
plt.close('all')

n = 8
#sub_dim = [2*2,4]
sub_dim = [2*2,4*2]

plt.figure(figsize=(18,10))
plt.subplots_adjust(wspace = 0.2, hspace = 0.5)

i_file = np.random.randint(8000, size = int(n/2))
#i_file[0] =  4225 #Unwanted peeks
#i_file[1] =  1869 #Unwanted peeks
#i_file[2] =  4733 #Unwanted peeks

AF_file_list = []
SIN_file_list = []

for i_data in range(n):
    if i_data < n/2:        
#        main_path = 'C:\Hinkelstien/data/FILTERED/sinus_rhythm_8k/'
        main_path = '/vol/hinkelstn/data/FILTERED/sinus_rhythm_8k/'        
        plt_color = 'b'
        plt_title = 'Sinus'
        i_f = i_data
        i_base = 0
#        i_plt_ch2 = 4+i_data+1
#        i_plt_x = 8+i_data+1
    else:
#        main_path = 'C:\Hinkelstien/data/FILTERED/atrial_fibrillation_8k/'            
        main_path = '/vol/hinkelstn/data/FILTERED/atrial_fibrillation_8k/'    
        plt_color = 'r'
        plt_title = 'AF'
        i_f = int(i_data-n/2)
        i_base = 8
        

    i_plt_ch1 = i_base+2*i_data+1
    i_plt_ch1x = i_base+2*i_data+2
    i_plt_ch2 = i_base+n+2*i_data+1
    i_plt_ch2x = i_base+n+2*i_data+2
    
        
    list_f = os.listdir(main_path)
    file = list_f[i_file[i_f]]
    path = main_path+file        
      
    w = wavio.read(path)
#    w.data = w.data[40000:50000,:]
#    w.data[:,1] = w.data[:,1]
  
    channel = w.data[:,0]
    plt.subplot(4,sub_dim[1],i_plt_ch1)
    #plt.figure(figsize=(8,6))
    plt.plot(channel,color = plt_color)
    plt.title(plt_title+', sample_id:'+str(i_file[i_f]))
    plt.grid()
    max_list, mean_max, thresh = wave_harsh_peaks(channel, silent  = True)
    list_p = np.where(channel>mean_max)    
    plt.scatter(list_p, channel[list_p], color = 'g')
    
    data_masked = scipy.stats.mstats.winsorize(channel, limits=[0, 0.001])
    plt.plot(data_masked,color = 'y')
    
    plt.subplot(4,sub_dim[1],i_plt_ch1x)     
    
    
    plt.grid()
    plt.scatter(range(len(max_list)), max_list)
    plt.scatter(range(len(max_list)), mean_max*np.ones(len(max_list)), color = 'g')
    plt.scatter(range(len(max_list)), thresh*np.ones(len(max_list)), color = 'r')

    channel = w.data[:,1]    
    plt.subplot(4,sub_dim[1],i_plt_ch2)
    plt.plot(w.data[:,1],color = plt_color)
    plt.xlabel('samples')
    plt.grid()
    max_list, mean_max, thresh = wave_harsh_peaks(channel, silent  = True)
    list_p = np.where(channel>mean_max)    
    plt.scatter(list_p, channel[list_p], color = 'g')
    data_masked = scipy.stats.mstats.winsorize(channel, limits=[0, 0.001])
    plt.plot(data_masked,color = 'y')
    
    plt.subplot(4,sub_dim[1],i_plt_ch2x)
    max_list, mean_max, thresh = wave_harsh_peaks(w.data[:,1], silent  = True)
    plt.grid()
    plt.scatter(range(len(max_list)), max_list)
    plt.scatter(range(len(max_list)), mean_max*np.ones(len(max_list)), color = 'g')
    plt.scatter(range(len(max_list)), thresh*np.ones(len(max_list)), color = 'r')
    
    
    
    
#    plt.figure()
#    plt.scatter(range(705),w.data[34745:35450,0],color = plt_color)
  
#    if i_data <n/2:
#        AF_file_list.append(file)
#    else:
#        SIN_file_list.append(file)
#AF_file_list = [{}]        
assert 1== 61090


# %%================= individual files  
    
           
i_file = 6587
#i_file = np.random.randint(8000, size = 1).item()

#i_class = 0 #0:normal  1:atrial
i_class = np.random.randint(2, size = 1)+1    
    
#path = '/data/BMBF/sample/filtered atrial waves/8aae6985-c0b9-41d4-ac89-9f721b8019d2.wav'
if i_class==0:
#    main_path = 'C:\Hinkelstien/data/FILTERED/sinus_rhythm_8k/'
    main_path = '/vol/hinkelstn/data/FILTERED/sinus_rhythm_8k/'    
    plt_color = 'b'
    plt_title = 'normal sinus rhythm'
else:
#    main_path = 'C:\Hinkelstien/data/FILTERED/atrial_fibrillation_8k/'    
    main_path = '/vol/hinkelstn/data/FILTERED/atrial_fibrillation_8k/'    
    plt_color = 'r'
    plt_title = 'Atrial Fibrilation'
    
list_f = os.listdir(main_path)
file = list_f[i_file]
path = main_path+file

#==============================
w = wavio.read(path)
#w.data = w.data[30000:50000,:]

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
plt.grid()

#plt.figure()
#plt.scatter(range(705),w.data[34745:35450,0],color = plt_color)

my_data_classes.wave_harsh_peaks(w.data[:,0])

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
