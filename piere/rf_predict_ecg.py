#import argparse
import time
import glob
import os
import pickle
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import wavio
from scipy import stats



def extract_stable_part(data, win_size = 2048, stride = 512):
    assert(win_size % stride == 0)

    offset = len(data) % stride
    indices = np.arange(offset, len(data) - (stride - 1), stride)
    maxs = np.zeros((len(indices), 2))
    mins = np.zeros((len(indices), 2))
    over = np.zeros_like(indices)
    undr = np.zeros_like(indices)
    for i, idx in enumerate(indices):
        win = data[idx:idx + stride]
        maxs[i] = win.max(axis=0)
        mins[i] = win.min(axis=0)
        over[i] = (win < 100).sum()
        undr[i] = (win > 3900).sum()

    str_in_win = win_size // stride
    min_diff = np.inf
    best = 0
    for i, idx in enumerate(indices[:-(str_in_win - 1)]):
        chan_diff = maxs[i:i + str_in_win].max(axis=0) - mins[i:i + str_in_win].min(axis=0)
        if (chan_diff > 0).all():
            diff = chan_diff.sum() + over[i:i + str_in_win].sum() + undr[i:i + str_in_win].sum()
            if diff < min_diff:
                min_diff = diff
                best = idx
    
    return data[best:best + win_size]

def check_constant(channel_data):
    first = channel_data[0]
    for i in range(1, len(channel_data)):
        if channel_data[i] != first:
            return False
    return True


def tree_summary(estimator):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold


    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
        "the following tree structure:"
        % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %.3f else to "
                "node %s."
                % (node_depth[i] * "\t",
                    i,
                    children_left[i],
                    feature[i],
                    threshold[i],
                    children_right[i],
                    ))

#def main(args):
#========paths:
features = "./Features.xlsx"
#ecg_dir = '/vol/hinkelstn/data/'+'ecg_8k_wav'+'/sinus_rhythm_8k/'
ecg_dir = '/vol/hinkelstn/data/'+'ecg_8k_wav'+'/atrial_fibrillation_8k/'
#rf = "./rf9_nf_cv0_pickle.p"
#rf = "./2c_rf20_0mean_cv10.p"
#rf = "./2c_rf15_0mean_cv16.p"
rf = "./2c_rf15_0mean_cv22.p"

os.chdir('/home/bhossein/BMBF project/code_repo/piere')

#========parameters:
verbose = False
window = 2048 
stride = 512
sub = 8
#rf_seed = 1
#rf_size = 10

xls = pd.read_excel(features)
#rf = pickle.load(open(rf, "rb"))
rf_model = pickle.load(open(rf, "rb"))
rf = rf_model['rf']
maxX = rf_model['maxX']
maxint = rf_model['maxint']
norm = rf_model['norm']
m_trn = norm[0]
v_trn = norm[1]


estimator = rf.estimators_[0] 

help(tree._tree.Tree)


tree_summary(estimator)

print(estimator.tree_.feature) # feature array of one DT from the RF model


print(estimator.tree_.max_depth)
internal = [[estimator.tree_.feature,
             estimator.tree_.threshold,
             estimator.tree_.children_left,
             estimator.tree_.children_right,
             np.argmax(estimator.tree_.value[estimator.tree_.feature == -2][:,0], axis=-1)]
             for estimator in rf.estimators_]
    
#%%    testing te trained RF    

pos = 0
neg = 0
tp = 0
fp = 0

ecg_count = 0
for ecg_file in glob.glob(os.path.join(ecg_dir, "*.wav")):    
    ecg_count +=1

    if ecg_count % 1000 ==0:
        print(ecg_count)
    features = xls.loc[xls["Filename"] == os.path.basename(ecg_file)]
    label = features["Output 2"].values
   
    features = features.iloc[:,[0, 2, 5, 6, 7, 8, 9, 10, 11, 12]]    

    ecg = wavio.read(ecg_file)
    ecg0 = ecg
#    ecg = extract_stable_part(ecg.data, window, stride)
    
    ecg = ecg.data[len(ecg.data)-stride-window:len(ecg.data)-stride]
    
    if check_constant(ecg[:,0]) or check_constant(ecg[:,1]):
        ecg = extract_stable_part(ecg0.data, window, stride)
        print(ecg_file)
        
    ecg = ecg[0::sub,:]    
#    ecg = stats.zscore(ecg, axis = 0, ddof = 1)
    m = np.mean(ecg, axis=0)
    ecg  = ecg-m
    ecg[np.isnan(ecg)]=0

    features = np.append(features, np.max(ecg[:,0], axis=0))
    features = np.append(features, np.max(ecg[:,1], axis=0))  
    features = np.append(features, np.min(ecg[:,0], axis=0))
    features = np.append(features, np.min(ecg[:,1], axis=0))

    features = (features - m_trn) / v_trn
    
    
    features = [features[i]/maxX[i]*maxint for i in range(len(features))]
    features = np.array(features).astype(int)
  
       
    X = features.reshape(1, -1)
    pred = rf.predict(X)
    pred_trees = [rf.estimators_[i].tree_.predict(X.astype('float32')) for i in range(len(rf))]
    tree_decicsions = np.argmax(pred_trees,axis = 2)

    if label == 0:
        neg += 1
        if pred == 1:
            fp += 1
    else:
        pos += 1
        if pred == 1:
            tp += 1
    
    if verbose:        
        print("{}: {}".format(os.path.basename(ecg_file), "Sinus" if pred == 0 else "AF"))
        
print("TP: {}/{}".format(tp, pos))
print("FP: {}/{}".format(fp, neg))

if pos>0:
    print('tp rate:',tp/pos*100)
else:
    print('fp rate:',fp/neg*100)
