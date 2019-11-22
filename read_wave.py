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

os.chdir('/home/bhossein/BMBF project/code')
#==============================
#plt.close('all')

#i_file = 2
i_file = np.random.randint(8000, size = 1)
#i_class = 2 #1:normal  2:atrial
i_class = np.random.randint(2, size = 1)+1

#path = '/data/BMBF/sample/filtered atrial waves/8aae6985-c0b9-41d4-ac89-9f721b8019d2.wav'
if i_class==1:
    main_path = '/vol/hinkelstn/data/FILTERED/sinus_rhythm_8k/'
    plt_color = 'b'
    plt_title = 'normal sinus rhythm'
else:
    main_path = '/vol/hinkelstn/data/FILTERED/atrial_fibrillation_8k/'    
    plt_color = 'r'
    plt_title = 'Atrial Fibrilation'
    
list_f = os.listdir(main_path)
file = list_f[i_file[0]]
path = main_path+file

#==============================
#rate = 20  # samples per second
#T = 3         # sample duration (seconds)
#f = 2.0     # sound frequency (Hz)
#t = np.linspace(0, T, T*rate, endpoint=False)
#x = np.sin(2*np.pi * f * t)
#wavio.write("sine24.wav", x, rate, sampwidth=3)

#w = wavio.read("sine24.wav")
w = wavio.read(path)
#plt.figure(figsize=(8,6))
#fig, ax = plt.subplots(2,1,1)




plt.figure(figsize=(8,6))
#plt.subplots(2,1,1)
plt.subplot(211)
#plt.figure(figsize=(8,6))
plt.plot(w.data[:,0],color = plt_color)
plt.title(plt_title+', sample_id:'+str(i_file))
plt.subplot(212)
plt.plot(w.data[:,1],color = plt_color)
plt.xlabel('samples')

#plt.figure(figsize=(8,6))
#plt.plot(t,x)
#
plt.figure()
#plt.plot(w.data[34745:35450,0],color = plt_color)
plt.scatter(range(705),w.data[34745:35450,0],color = plt_color)