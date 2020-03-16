#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DTW search

Created on Tue Mar 10 10:50:49 2020


@author: bhossein
"""


import numpy as np
import mass_ts as mts

import matplotlib.pyplot as plt

import wavio
import os
os.chdir('/home/bhossein/BMBF project/code_repo')

#%% Read ECG
FIR = 0
          
#i_file = 1000
i_file = np.random.randint(8000, size = 1).item()
#i_file = 4275  #MAIN QUERY



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
print(file)
#file = IDs_Dfn_Ktp[0]
#file = IDs[2254]
#file = "dadb672d-8c3a-4811-9cab-f8de7c309987.wav"
#file = "ecbf5c09-c0ef-426f-b83e-31afbeac899e.wav"
#file = "e9723cc4-6fb4-43f2-a4aa-ad68ee160e18.wav"

path = main_path+file

#==============================
w = wavio.read(path)
data = np.transpose(w.data)
x = data[1,:] - data[0,:]
data = np.row_stack((data,x))


fig, axes = plt.subplots(3, 1, sharex=True)

plt_color = 'b'
if FIR == 1:
    plt_title = 'FIR data : ' + plt_title
#    plt_color = 'b'
else:
    plt_title = 'Raw data : ' + plt_title
#    plt_color = 'r'
    
for i_ch in range(3):
#    i_ax = 0
    i_ax = i_ch
    channel = data[i_ch,:]
    
    axes[i_ax].plot(channel,color = plt_color)
    if i_ch == 0:
#            axes[i_ax].set_title(plt_title+', sample_id:'+str(i_file))
        axes[i_ax].set_title(plt_title+' file:'+ file)

    axes[i_ax].grid()
        
    if i_ch ==2:
        axes[i_ax].set_xlabel('Time steps')

#%% ============== manual motifs
#data0 = data.copy()

i_chh = 1
k = 10
exclude_zone = 300
t_s = 57000
t_step = 1000
t = range(t_s,t_s+t_step)
quary = data0[i_chh-1,t]
target = data0[i_chh-1,:]

distances = mts.mass2(target, quary)
#distances = np.array([abs(i) for i in distances])
#distances.sort()

found = mts.top_k_motifs(distances, k, exclude_zone)
indices = np.array(found)
distances = distances[found] 

#indices, distances = mts.mass2_batch(target, quary, 1000, top_matches = k)

i_sort = np.argsort(distances)
distances, indices = [t[i_sort] for t in (distances, indices)]
distances = [abs(i) for i in distances]


plt.figure()
plt.subplot(2,1,1)
plt.plot(quary)
plt.subplot(2,1,2)
plt.plot(target)
for ind in indices:
#        plt.subplot(2,1,2)
    t = range(ind,ind+int(1*len(quary)))
    plt.plot(t, target[t], color = 'g')

t = range(indices[0],indices[0]+int(1*len(quary)))
plt.plot(t, target[t], color = 'r')

t = range(indices[k-1],indices[k-1]+int(1*len(quary)))
plt.plot(t, target[t], color = 'y')

t = range(indices[1],indices[1]+int(1*len(quary)))
plt.plot(t, target[t], color = 'c')

print(distances[:5])
#%% === find quary in another ECG
#data2 = data.copy()
#k = 60
target = data2[i_chh-1,:]

distances2 = mts.mass2(target, quary)
#distances = np.array([abs(i) for i in distances])
#distances.sort()

found = mts.top_k_motifs(distances2, k, exclude_zone)
indices2 = np.array(found)
distances2 = distances2[found] 

#indices2, distances2 = mts.mass2_batch(
#    target, quary, 1000, top_matches = k)

i_sort = np.argsort(distances2)
distances2, indices2 = [t[i_sort] for t in (distances2, indices2)]
distances2 = [abs(i) for i in distances2]

plt.figure()
plt.subplot(2,1,1)
plt.plot(quary)
plt.subplot(2,1,2)
plt.plot(target)
for ind in indices2:
#        plt.subplot(2,1,2)
    t = range(ind,ind+int(1*len(quary)))
    plt.plot(t, target[t], color = 'g')

#t = range(indices[0],indices[0]+int(1*len(quary)))
#plt.plot(t, target[t], color = 'r')

t = range(indices2[k-1],indices2[k-1]+int(1*len(quary)))
plt.plot(t, target[t], color = 'y')

t = range(indices2[0],indices2[0]+int(1*len(quary)))
plt.plot(t, target[t], color = 'c')

print(distances2[:5])