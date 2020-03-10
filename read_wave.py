#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:28:34 2019

@author: bhossein
"""
import numpy as np
import wavio
import os
os.chdir('/home/bhossein/BMBF project/code_repo')

import matplotlib.pyplot as plt
#import scipy

from my_data_classes import create_datasets, create_loaders, read_data, create_datasets_file, smooth, wave_harsh_peaks, wave_harsh_peaks_all
import my_data_classes



#%%================== clipping
plt.close('all')
FIR = 0
n = 8
#sub_dim = [2*2,4]
sub_dim = [2*2,4*2]

th_ratio = 1.1

plt.figure(figsize=(18,10))
plt.subplots_adjust(wspace = 0.2, hspace = 0.5)

i_file = np.random.randint(8000, size = int(n/2))
#i_file[0] =  4225 #Unwanted peeks
#i_file[1] =  1869 #Unwanted peeks
#i_file[2] =  4733 #Unwanted peeks

AF_file_list = []
SIN_file_list = []

filter_path = '/FILTERED' if FIR else '/ecg_8k_wav'

path_data = '/vol/hinkelstn/data'+filter_path+'/sinus_rhythm_8k/'  
path_data = np.append(path_data,'/vol/hinkelstn/data'+filter_path+'/atrial_fibrillation_8k/')


for i_data in range(n):
    if i_data < n/2:        
#        main_path = 'C:\Hinkelstien/data/FILTERED/sinus_rhythm_8k/'
        main_path = path_data[0]
        plt_color = 'b'
        plt_title = 'Sinus'
        i_f = i_data
        i_base = 0
#        i_plt_ch2 = 4+i_data+1
#        i_plt_x = 8+i_data+1
    else:
#        main_path = 'C:\Hinkelstien/data/FILTERED/atrial_fibrillation_8k/'            
        main_path = path_data[1]
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
    
    trimm_out = wave_harsh_peaks(channel, ax  = 'silent', t_base = 3000)
    max_list, mean_max, trimmed_t = (trimm_out[0],trimm_out[1],trimm_out[4])
    list_p = np.where(channel>mean_max)    
    plt.scatter(list_p, channel[list_p], color = 'g')    
    data_masked = channel[trimmed_t]
    #data_masked = scipy.stats.mstats.winsorize(channel, limits=[0, 0.001])
    plt.plot(trimmed_t, data_masked,color = 'y')
    plt.grid()
    
        
    plt.subplot(4,sub_dim[1],i_plt_ch1x)         
    plt.grid()
    plt.scatter(range(len(max_list)), max_list)
    plt.scatter(range(len(max_list)), mean_max*np.ones(len(max_list)), color = 'g')
#    plt.scatter(range(len(max_list)), thresh*np.ones(len(max_list)), color = 'r')

    channel = w.data[:,1]    
    plt.subplot(4,sub_dim[1],i_plt_ch2)
    plt.plot(w.data[:,1],color = plt_color)
    plt.xlabel('samples')
    plt.grid()
    
    trimm_out = wave_harsh_peaks(channel, ax  = 'silent', t_base = 3000, th_ratio= th_ratio)
    max_list, mean_max, trimmed_t = (trimm_out[0],trimm_out[1],trimm_out[4])
    list_p = np.where(channel>mean_max)    
    plt.scatter(list_p, channel[list_p], color = 'g')    
    data_masked = channel[trimmed_t]
    #data_masked = scipy.stats.mstats.winsorize(channel, limits=[0, 0.001])
    plt.plot(trimmed_t, data_masked,color = 'y')
    plt.grid()
    
    plt.subplot(4,sub_dim[1],i_plt_ch2x)    
    plt.grid()
    plt.scatter(range(len(max_list)), max_list)
    plt.scatter(range(len(max_list)), mean_max*np.ones(len(max_list)), color = 'g')
#    plt.scatter(range(len(max_list)), thresh*np.ones(len(max_list)), color = 'r')
    
    
    
    
#    plt.figure()
#    plt.scatter(range(705),w.data[34745:35450,0],color = plt_color)
  
#    if i_data <n/2:
#        AF_file_list.append(file)
#    else:
#        SIN_file_list.append(file)
#AF_file_list = [{}]        
assert 1== 61090


# %%================= individual files  
FIR = 0
thresh_rate =1.21
           
#i_file = 1000
#i_file = np.random.randint(8000, size = 1).item()
i_file = 7129

i_class =0 #0:normal  1:atrial
#i_class = np.random.randint(2, size = 1)+1    
    
#path = '/data/BMBF/sample/filtered atrial waves/8aae6985-c0b9-41d4-ac89-9f721b8019d2.wav'

filter_path = '/FILTERED' if FIR else '/ecg_8k_wav'
path_data = '/vol/hinkelstn/data'+filter_path+'/sinus_rhythm_8k/'  
path_data = np.append(path_data,'/vol/hinkelstn/data'+filter_path+'/atrial_fibrillation_8k/')


if i_class==0:
    main_path = path_data[0]
    plt_color = 'b'
    plt_title = 'normal sinus rhythm'
else:
    main_path = path_data[1]
    plt_color = 'r'
    plt_title = 'Atrial Fibrilation'
    
list_f = os.listdir(main_path)
file = list_f[i_file]
#file = IDs_Dfn_Ktp[0]
#file = IDs[2254]
#file = "dadb672d-8c3a-4811-9cab-f8de7c309987.wav"
#file = "ecbf5c09-c0ef-426f-b83e-31afbeac899e.wav"
#file = "e9723cc4-6fb4-43f2-a4aa-ad68ee160e18.wav"

path = main_path+file

#==============================
w = wavio.read(path)
#w.data = w.data[30000:50000,:]

#fig, axes = plt.subplots(3, 1, sharex=True)

#plt.figure(figsize=(8,6))
#plt.subplots(2,1,1)
#plt.subplot(311)
#plt.figure(figsize=(8,6))

trimm_out = wave_harsh_peaks_all(w.data, t_base = 3000, thresh_rate = thresh_rate)

fig, axes = plt.subplots(2, 1, sharex=True)

i_ax = 0
channel = w.data[:,0]

axes[i_ax].plot(channel,color = plt_color)
#plt.plot(w.data[:,0],color = plt_color)
axes[i_ax].set_title(plt_title+', sample_id:'+str(i_file))
#trimm_out = wave_harsh_peaks(channel, ax  = 'silent', t_base = 3000)


mean_max, list_t, trimmed_t = trimm_out
list_p = np.where(channel>mean_max[i_ax])    
#axes[i_ax].scatter(list_p, channel[list_p], color = 'g')    
data_masked = channel[trimmed_t]
#data_masked = scipy.stats.mstats.winsorize(channel, limits=[0, 0.001])
axes[i_ax].plot(trimmed_t, data_masked,color = 'g')
#axes[i_ax].scatter(trimmed_t, data_masked,color = 'g')
axes[i_ax].grid()

#plt.subplot(312)
i_ax = 1
channel = w.data[:,1]

axes[i_ax].plot(channel,color = plt_color)
#plt.title(plt_title+', sample_id:'+str(i_file))
#trimm_out = wave_harsh_peaks(channel, ax  = 'silent', t_base = 3000)
#mean_max, trimmed_t = (trimm_out[1],trimm_out[4])
#list_p = np.where(channel>mean_max)    
list_p = np.where(channel>mean_max[i_ax])
#axes[i_ax].scatter(list_p, channel[list_p], color = 'g')    
data_masked = channel[trimmed_t]
axes[i_ax].plot(trimmed_t, data_masked,color = 'g')
axes[i_ax].grid()

#i_ax = 2
#channel = w.data[:,1]-w.data[:,0]
#
#axes[i_ax].plot(channel,color = plt_color)
##plt.title(plt_title+', sample_id:'+str(i_file))
#trimm_out = wave_harsh_peaks(channel, ax  = 'silent', t_base = 3000)
#mean_max, trimmed_t = (trimm_out[1],trimm_out[4])
#list_p = np.where(channel>mean_max)    
#axes[i_ax].scatter(list_p, channel[list_p], color = 'g')    
#data_masked = channel[trimmed_t]
#axes[i_ax].plot(trimmed_t, data_masked,color = 'y')
##plt.xlabel('samples')
#axes[i_ax].grid()

axes[i_ax].set_xlabel('Time steps')

my_data_classes.wave_harsh_peaks(w.data[:,0], th_ratio =  thresh_rate )

my_data_classes.wave_harsh_peaks(w.data[:,1], th_ratio =  thresh_rate)

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


# %%========================== only plot
w = wavio.read(path)
w.data = w.data[0:5000,:]
t = np.arange(0,len(w.data)) / w.rate

fig, axes = plt.subplots(3, 1, sharex=True)

#plt.subplot(311)
#plt.figure(figsize=(8,6))
axes[0].plot(t, w.data[:,0],color = plt_color)
#plt.plot(w.data[:,0],color = plt_color)
axes[0].set_title("normal sinus,   "+ file)
#axes[0].set_title(plt_title+', sample_id:'+str(i_file))
axes[0].grid()
axes[0].set_ylabel('Channel I')

axes[1].plot(t, w.data[:,1],color = plt_color)
axes[1].grid()
axes[1].set_ylabel('Channel III')

axes[2].plot(t, w.data[:,1]-w.data[:,0],color = plt_color)
plt.xlabel('Seconds')
axes[2].grid()
axes[2].set_ylabel('Channel III - Channel I')

#%%================== Raw Vs. FIR
#plt.close('all')
thresh_rate =1.21
           
#i_file = 1000
#i_file = np.random.randint(8000, size = 1).item()
i_file = 5840

i_class = 1 #0:normal  1:atrial
#i_class = np.random.randint(2, size = 1)+1    
    
#plt.figure(figsize=(18,10))
#plt.subplots_adjust(wspace = 0.2, hspace = 0.5)

fig, axes = plt.subplots(4, 1, sharex=True)

for i in range(2):
    FIR = i
    filter_path = '/FILTERED' if FIR else '/ecg_8k_wav'
    path_data = '/vol/hinkelstn/data'+filter_path+'/sinus_rhythm_8k/'  
    path_data = np.append(path_data,'/vol/hinkelstn/data'+filter_path+'/atrial_fibrillation_8k/')
    
    
    if i_class==0:
        main_path = path_data[0]
#        plt_color = 'b'
        plt_title = 'Normal,  '
    else:
        main_path = path_data[1]
#        plt_color = 'r'
        plt_title = 'AF,  '
        
    if i == 0:
        list_f = os.listdir(main_path)
        file = list_f[i_file]    
    print(file)
    path = main_path+file
    
    #==============================    
    w = wavio.read(path)
    
    if FIR == 1:
        plt_title = 'FIR data : ' + plt_title
        plt_color = 'b'
    else:
        plt_title = 'Raw data : ' + plt_title
        plt_color = 'r'
        
    for i_ch in range(2):
#    i_ax = 0
        i_ax = i_ch + i*2
        channel = w.data[:,i_ch]
        
        axes[i_ax].plot(channel,color = plt_color)
        if i_ch == 0:
#            axes[i_ax].set_title(plt_title+', sample_id:'+str(i_file))
            axes[i_ax].set_title(plt_title+' file:'+ file)
    
        axes[i_ax].grid()
        
        #plt.subplot(312)
#        i_ax = 1
#        channel = w.data[:,i_ch]
        
#        axes[i_ax].plot(channel,color = plt_color)
#        axes[i_ax].grid()
        if i_ch ==1 and i ==1:
            axes[i_ax].set_xlabel('Time steps')


#plt.savefig('{}.png'.format(file[:-4]))
assert 1== 61090

# %%============================= bad data list
bad_list = []
bad_files = []

IDs = []
main_path = '/vol/hinkelstn/data/FILTERED/atrial_fibrillation_8k/'    
IDs.extend(os.listdir(main_path))
IDs = os.listdir(main_path)
main_path = '/vol/hinkelstn/data/FILTERED/sinus_rhythm_8k/'
IDs.extend(os.listdir(main_path))

target = np.ones(16000)
target[0:8000]=0

for i_ID, ID in enumerate(IDs):
#    ID = IDs[i_ID]
#    print('sample: %d , time: %5.2f (s)' % (i_ID, millis2-millis))
#        print(millis2-millis)
#    millis = (time.time())
#    pickle.dump({'i_ID':i_ID},open("read_data_i_ID.p","wb"))
    if i_ID % 1000 == 0:
#            pickle.dump({'i_ID':i_ID},open("read_data_i_ID.p","wb"))
        print(i_ID)
    y = target[i_ID]
#    assert y <= target.max()
    # Load data and get label
    if y == 0:
        main_path = '/vol/hinkelstn/data/FILTERED/atrial_fibrillation_8k/'
#                        main_path = '/data/bhosseini/hinkelstn/FILTERED/atrial_fibrillation_8k/'
    else:
        main_path = '/vol/hinkelstn/data/FILTERED/sinus_rhythm_8k/'
#                        main_path = '/data/bhosseini/hinkelstn/FILTERED/sinus_rhythm_8k/'            
    path = main_path+ID
    w = wavio.read(path)
    if len(w.data) < 5000:
        bad_list = np.append(bad_list,i_ID)
        bad_files = np.append(bad_files,ID)
        