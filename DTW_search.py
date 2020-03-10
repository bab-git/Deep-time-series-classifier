#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DTW search

Created on Tue Mar 10 10:50:49 2020


@author: bhossein
"""


import numpy as np
import mass_ts as mts

import wavio
import os
os.chdir('/home/bhossein/BMBF project/code_repo')

FIR = 0
          
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

fig, axes = plt.subplots(2, 1, sharex=True)

plt_color = 'b'
if FIR == 1:
    plt_title = 'FIR data : ' + plt_title
#    plt_color = 'b'
else:
    plt_title = 'Raw data : ' + plt_title
#    plt_color = 'r'
    
for i_ch in range(2):
#    i_ax = 0
    i_ax = i_ch
    channel = w.data[:,i_ch]
    
    axes[i_ax].plot(channel,color = plt_color)
    if i_ch == 0:
#            axes[i_ax].set_title(plt_title+', sample_id:'+str(i_file))
        axes[i_ax].set_title(plt_title+' file:'+ file)

    axes[i_ax].grid()
        
    if i_ch ==1 and i ==1:
        axes[i_ax].set_xlabel('Time steps')


# ============== manual motifs

k = 10
indices, distances = mts.mass2_batch(
    robot_dog, carpet_walk, 1000, top_matches = k)
min_dist_idx = np.argmin(distances)
min_idx = indices[min_dist_idx]
max_idx = indices[k-1]

plt.figure()
plt.subplot(2,1,1)
plt.plot(carpet_walk)
plt.subplot(2,1,2)
plt.plot(robot_dog)
for ind in indices:
#        plt.subplot(2,1,2)
    t = range(ind,ind+int(1*len(carpet_walk)))
    plt.plot(t, robot_dog[t], color = 'g')

t = range(min_idx,min_idx+int(1*len(carpet_walk)))
plt.plot(t, robot_dog[t], color = 'r')

t = range(max_idx,max_idx+int(1*len(carpet_walk)))
plt.plot(t, robot_dog[t], color = 'y')