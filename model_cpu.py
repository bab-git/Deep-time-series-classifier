# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:48:24 2020

@author: babak
"""
from default_modules import *

#%%======
list_files = os.listdir(result_dir)

for file_name in list_files:
    if all(i in file_name for i in ('train_','_best.pth')):
        model_path = result_dir+file_name
        try:
            model = pickle.load(open(model_path, 'rb'))
        except:
            print(model_path+' not saved as model')
            continue
        if not getattr(model,'parameters',None):
            continue
        
        if next(model.parameters()).is_cuda:
            model = model.to('cpu')
            pickle.dump(model,open(model_path,'wb'))
            print(model_path+' is saved on CPU')
        else:
            print(model_path+' is already saved on CPU')
            
print('Finished the conversoin')
    