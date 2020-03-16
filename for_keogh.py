#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:52:55 2020

@author: bhossein
"""
import csv
import wavio
#%%================== Raw Vs. FIR
#plt.close('all')
           
#i_file = 1000
i_file_set = np.random.randint(8000, size = 10)
#i_file = 5840
file_set = []
#i_class = 0 #0:normal  1:atrial
#i_class = np.random.randint(2, size = 1)+1    
    
#plt.figure(figsize=(18,10))
#plt.subplots_adjust(wspace = 0.2, hspace = 0.5)

#fig, axes = plt.subplots(4, 1, sharex=True)
save_po = 'Keogh/'
FIR_count = 0
for i in range(2):
    FIR = i 
    save_r = save_po +'FIR' if FIR else save_po +'raw'
    filter_path = '/FILTERED' if FIR else '/ecg_8k_wav'
    path_data = '/vol/hinkelstn/data'+filter_path+'/sinus_rhythm_8k/'  
    path_data = np.append(path_data,'/vol/hinkelstn/data'+filter_path+'/atrial_fibrillation_8k/')
    
    
    for i_class in range(2):
        if i_class==0:
            main_path = path_data[0]
            plt_title = 'sin,  '
            save_p = save_r + '/class1/'
        else:
            main_path = path_data[1]
    #        plt_color = 'r'
            plt_title = 'AF,  '       
            save_p = save_r + '/class2/'
            
        list_f = os.listdir(main_path)
            
        for i_file in i_file_set:
            if FIR == 0:
                file = list_f[i_file]        
                file_set = np.append(file_set, file)
            else:
                
                file = file_set[FIR_count]
                FIR_count += 1
                
            path = main_path+file
            w = wavio.read(path)
            w = w.data[:60000,:].transpose()
            csv_file = save_p+file[:-3]+'csv'
            f = open(csv_file, 'w')
            with f:
                writer = csv.writer(f)
                for row in w:
                    writer.writerow(row)