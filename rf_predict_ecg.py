import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
import wavio
from scipy import stats

def extract_stable_part(data, win_size = 2048, stride = 512, sub = 1):
    data0 = data
    data = data0[::sub]
    win_size = int(np.floor(win_size / sub))
    stride = int(np.floor(stride /sub))
    offset = len(data) % stride
    indices = np.arange(offset, len(data) - (win_size - 1), stride)
    min_diff = np.inf
    best = 0
    for idx in indices:
        win = data[idx:idx + win_size]
        diff = np.sum(np.max(win, axis=0) - np.min(win, axis=0)) + (win < 100).sum() + (win > 3900).sum()
        if diff < min_diff:
            min_diff = diff
            best = idx
    best = best * sub
    win_size = win_size *sub
    stride = stride*sub
    return data0[best:best + win_size]

#def main(args):
#========paths:
features = "./Features.xlsx"
#ecg_dir = '/vol/hinkelstn/data/'+'ecg_8k_wav'+'/sinus_rhythm_8k/'
ecg_dir = '/vol/hinkelstn/data/'+'ecg_8k_wav'+'/atrial_fibrillation_8k/'
rf = "./rf9_nf_cv0_pickle.p"

#========parameters:
verbose = False
window = 2048 
stride = 512
sub = 1

xls = pd.read_excel(features)
rf = pickle.load(open(rf, "rb"))

#print(rf.estimators_[0].tree_.feature) # feature array of one DT from the RF model

pos = 0
neg = 0
tp = 0
fp = 0

ecg_count = 0
for ecg_file in glob.glob(os.path.join(ecg_dir, "*.wav")):
    ecg_count +=1
#    t1 = time.time()
#    print(ecg_file)
    features = xls.loc[xls["Filename"] == os.path.basename(ecg_file)]
    label = features["Output 2"].values
    features = features.iloc[:,[0, 2, 5, 6, 7, 8, 9, 11, 12]]

    ecg = wavio.read(ecg_file)
#    ecg = ecg0
    ecg = extract_stable_part(ecg.data, window, stride, sub)
    ecg = stats.zscore(ecg, axis = 0, ddof = 1)
    ecg[np.isnan(ecg)]=0

    features = np.append(features, np.max(ecg, axis=0))
    features = np.append(features, np.min(ecg, axis=0))
    
    pred = rf.predict(features.reshape(1, -1))

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
        
#    if ecg_count % 1000 == 0 :
#        print("TP: {}/{}".format(tp, pos))
#        print(fp/neg*100)
#        print(tp/pos*100)

#    t2 = time.time()
#    print(t2-t1)
print("TP: {}/{}".format(tp, pos))
print("FP: {}/{}".format(fp, neg))

#if pos:
#    print(tp/pos*100)
#else:
#    print(fp/neg*100)
    


#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument("-f", "--features",         default="./Features.xlsx",        help="Path to Features.xlsx (default: ./Features.xlsx)")
#    parser.add_argument("-r", "--rf",               default="./rf9_nf_cv0_pickle.p",  help="Path to RF model (default: ./rf9_nf_cv0.p)")
#    parser.add_argument("-w", "--window", type=int, default=2048,                     help="Size of the extracted stable part (default: 2048)")
#    parser.add_argument("-s", "--stride", type=int, default=512,                      help="Step size for stable part extraction (default: 512)")
#    parser.add_argument("ecg_dir",                                                    help="Path to ECG directory")
#    args = parser.parse_args()
#    main(args)
