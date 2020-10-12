#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:46:56 2020

@author: bhossein
"""
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt
import numpy as np
#%% 
X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
#%%
qbit = 1
X = np.round(qbit*X)/qbit

n_estimators = 4
clf = RandomForestClassifier(n_estimators = n_estimators, max_depth=2, random_state=0)
clf.fit(X, y)
RandomForestClassifier(...)
print(clf.predict([[0, 0, 0, 0]]))

out = clf.predict(X)
correct = np.where(out==y)
acc = len(correct[0])/len(y)*100
print(acc)

count = 0
for i in range(n_estimators):
    count += clf.estimators_[i].tree_.node_count
    print(clf.estimators_[i].tree_.node_count)
print(count) 
a = clf.estimators_[0] 
#plt.figure(10,10) 
tree.plot_tree(a) 

count = 0
for i in range(n_estimators):
    count += clf.estimators_[i].tree_.node_count
    print(clf.estimators_[i].tree_.node_count)
print(count)

#%%
#save_name= 'rf9_nf_cv1.p'

save_name= '3c_rf6_nf_norm_k2_cv1.p'
result_dir = 'results/'
rf_model = joblib.load(result_dir+save_name, mmap_mode='r')
#model = pickle.load(open(result_dir+'train_'+save_name+'_best.pth', 'rb'))
#
#df1 = pd.DataFrame(a,
#                   columns=['features', 'threshold','child_left','child_right'])
#df1 = pd.DataFrame(list(a),
#                   columns=['features', 'threshold','child_left','child_right'])



for j in range(len(rf_model)):
    x = np.zeros(len(rf_model[j][0]))
    b = np.where(rf_model[j][0]==-2)[0]
    l = rf_model[j][4]
    for i in range(len(b)):
        x[b[i]] = l[i]
    rf_model[j][4] = x

columns=['Features', 'Threshold','Child_left','Child_right', 'Label']
with xlsxwriter.Workbook('test.xlsx') as workbook:
    for i_tr in range(len(rf_model)):
        worksheet = workbook.add_worksheet(name = 'tree'+str(i_tr))
        worksheet.write_row(0,0, columns)
        tree_data = rf_model[i_tr]
        for row_num, data in enumerate(tree_data):
            worksheet.write_column(1,row_num, data)