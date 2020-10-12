#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 12:06:23 2020

@author: bhossein
"""

import pandas as pd
import xlsxwriter

columns=['Mean-ch1', 'std-ch1','min-ch1','max-ch1','Mean-ch2', 'std-ch2','min-ch2','max-ch2']
with xlsxwriter.Workbook('test.xlsx') as workbook:
#    for i_tr in range(raw_x.shape[0]):
    worksheet = workbook.add_worksheet(name = 'ECG data')
    worksheet.write_row(0,0, columns)
#    tree_data = rf_model[i_tr]
#    for row_num in range(2):
    for row_num in range(raw_x.shape[0]):        
        data = [[raw_x[row_num,i,:].mean(), raw_x[row_num,i,:].std(), raw_x[row_num,i,:].min(), raw_x[row_num,i,:].max()] for i in range(2)]
        data = list(np.array(data).reshape(1,-1)[0])
        worksheet.write_row(row_num+1,0,data)
