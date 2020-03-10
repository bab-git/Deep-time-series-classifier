#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:04:12 2020

@author: bhossein
"""
from default_modules import *
import ecg2wav
#%% ====================

path_data = '/vol/hinkelstn/data/ecg_8k/sinus_rhythm_8k/'    
out_path_data = '/vol/hinkelstn/data/ecg_8k_wav/sinus_rhythm_8k/'    

path_data = np.append(path_data,'/vol/hinkelstn/data/ecg_8k/atrial_fibrillation_8k/')
out_path_data = np.append(out_path_data,'/vol/hinkelstn/data/ecg_8k_wav/atrial_fibrillation_8k/')


for i in range(2):
    main_path = path_data[i]
    print(main_path)
    out_main_path = out_path_data[i]
    
    list_f = os.listdir(main_path)
    
    for i_file, file in enumerate(list_f):
        if i_file%100 == 0:
            print(i_file)
            
#        file = list_f[i_file[i_file]]
        in_path = main_path+file
        out_path = out_main_path+file
        ecg2wav.ecg2wave_conv(in_path, out_path)

print ('all ecg files are converted')      