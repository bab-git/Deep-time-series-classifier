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



#def extract_stable_part(data, win_size = 2048, stride = 512, sub = 1):
#    data0 = data
#    data = data0[::sub]
#    win_size = int(np.floor(win_size / sub))
#    stride = int(np.floor(stride /sub))
#    offset = len(data) % stride
#    indices = np.arange(offset, len(data) - (win_size - 1), stride)
#    min_diff = np.inf
#    best = 0
#    for idx in indices:
#        win = data[idx:idx + win_size]
#        diff = np.sum(np.max(win, axis=0) - np.min(win, axis=0)) + (win < 100).sum() + (win > 3900).sum()
#        if diff < min_diff:
#            min_diff = diff
#            best = idx
#    best = best * sub
#    win_size = win_size *sub
#    stride = stride*sub
#    return data0[best:best + win_size]


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
rf = "./rf9_nf_cv0_pickle.p"

os.chdir('/home/bhossein/BMBF project/code_repo/piere')

#========parameters:
verbose = False
window = 2048 
stride = 512
sub = 4
rf_seed = 1
rf_size = 10

#xls = pd.read_excel(features)
#rf = pickle.load(open(rf, "rb"))


#labels = np.argmax(estimator.tree_.value[:,0], axis=-1)
#labels_feat_2 = np.argmax(estimator.tree_.value[estimator.tree_.feature == -2][:,0], axis=-1)


features = np.loadtxt(fname = "Features.txt")
label_y = features[:,14]
X = features[:,:13]
maxint = np.iinfo(np.uint16).max 


X_test = X[tst_idx]
y_test = label_y[tst_idx]
X = X[trn_idx]
y_train = label_y[trn_idx]


#X, X_test, y_train, y_test = train_test_split(
#        X, label_y, test_size=0.2, random_state=2)

m_trn = X.mean(axis=0)
v_trn = X.std(axis=0)

#X = (X - m_trn) / v_trn
#X_test = (X_test - m_trn) / v_trn


maxX = [max(X[:,i]) for i in range(X.shape[1])]
X = np.array([X[:,i]/maxX[i]*maxint for i in range(X.shape[1])])
X = np.transpose(X).astype(int)


rf = RandomForestClassifier(random_state=1, n_estimators=10, ccp_alpha=4.0e-4, max_depth = 30)
rf.fit(X, y_train)

for i in range(len(rf)):
    rf.estimators_[i].tree_.threshold[:] = np.floor(rf.estimators_[i].tree_.threshold)




#%% RF detail

#estimator = rf.estimators_[0] 
#
#help(tree._tree.Tree)
#
#
#tree_summary(estimator)
#
#print(estimator.tree_.feature) # feature array of one DT from the RF model
#
#
#print(estimator.tree_.max_depth)
#internal = [[estimator.tree_.feature,
#             estimator.tree_.threshold,
#             estimator.tree_.children_left,
#             estimator.tree_.children_right,
#             np.argmax(estimator.tree_.value[estimator.tree_.feature == -2][:,0], axis=-1)]
#             for estimator in rf.estimators_]
    
#%%    testing te trained RF  
    
X_test = np.array([X_test[:,i]/maxX[i]*maxint for i in range(X_test.shape[1])])
X_test = np.transpose(X_test).astype(int)

pred = rf.predict(X_test)

TP = (pred[y_test==1] == 1).sum()/(y_test==1).sum()
FP = (pred[y_test==0] == 1).sum()/(y_test==0).sum()
print(TP)
print(FP)