# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 05:27:32 2020

@author: bhossein
"""
import matplotlib.pyplot as plt


import numpy as np

import pickle
#%%
#def evaluate(model, tst_dl, tst_idx, data_tag, thresh_AF = 3, device = 'cpu', acc_eval = False):
model.to(device)
input_shape = tuple(tst_dl.dataset.tensors[0].shape[1:3])
s = time.time()
model.eval()
correct, total , total_P, = 0, 0, 0
#    TP , FP = 0,0

batch = []
i_error = []
list_pred = []
with torch.no_grad():
    for i_batch, batch in enumerate(tst_dl):
        x_raw, y_batch = [t.to(device) for t in batch]
        list_x = list(range(i_batch*tst_dl.batch_size,min((i_batch+1)*tst_dl.batch_size,len(tst_dl.dataset))))
        # x_raw, y_batch = [t.to(device) for t in batch]
        # x_raw, y_batch = tst_ds.tensors
        # x_raw, y_batch = [t.to(device) for t in val_ds.tensors]
        out = model(x_raw)
        preds = out.argmax(dim=1)
        # preds = torch.sigmoid(out).squeeze().round()
        # preds = F.log_softmax(out, dim = 1).argmax(dim=1)
        # preds = F.log_softmax(out, dim = 1).argmax(dim=1).to('cpu')
        list_pred = np.append(list_pred,preds.cpu())
        # list_pred = np.append(list_pred,preds.tolist())
        total += y_batch.size(0)
        correct += (preds ==y_batch).sum().item()    
        # i_error = np.append(i_error,np.where(preds !=y_batch))
        i_error = np.append(i_error,[list_x[i] for i in np.where((preds !=y_batch).cpu())[0]])
        # TP += ((preds ==y_batch) & (1 ==y_batch)).sum().item()
        # total_P += (1 ==y_batch).sum().item()
        # FP += ((preds !=y_batch) & (0 ==y_batch)).sum().item()

elapsed = time.time() - s
print("{:>40}  {:<8.2f}".format("Elapsed time (seconds):", elapsed))
    
acc = correct / total * 100
#TP_rate = TP / total_P *100
#FP_rate = FP / (total-total_P) *100
#TP_rate = TP / (1 ==y_batch).sum().item() *100
#FP_rate = FP / (0 ==y_batch).sum().item() *100

print("{:>40}  {:<8.2f}".format("Accuracy on all windows of test data:", acc))

#if acc_eval:
#        return acc, list_pred
win_size = (data_tag==0).sum()
# thresh_AF = win_size /2
# thresh_AF = 3

list_ECG = np.unique([data_tag[i] for i in tst_idx])
#list_ECG = np.unique([data_tag[i] for i in tst_idx if target[i] == label])
#len(list_error_ECG)/8000*100

TP_ECG, FP_ECG , total_P, total_N = np.zeros(4)
idx_TP = []
idx_FP = []
list_pred_win = 100*np.ones([len(list_ECG), win_size])

TP_ECG_rate_hist = np.zeros([16,1],dtype=float)
FP_ECG_rate_hist = np.zeros([16,1],dtype=float)
for thresh_AF in range(1,16):
    print(thresh_AF)
    TP_ECG, FP_ECG , total_P, total_N = np.zeros(4)
    idx_TP = []
    idx_FP = []
    for i_row, i_ecg in enumerate(list_ECG):
        list_win = np.where(data_tag==i_ecg)[0]
        pred_win = [list_pred[tst_idx.index(i)] for i in list_win]
    #    print(pred_win)
        list_pred_win[i_row,:] = pred_win    
                            
        if i_ecg >= 8000:   #AF
            total_P += 1
            if (np.array(pred_win)==1).sum() >= thresh_AF:
                TP_ECG += 1
                idx_TP = np.append(idx_TP,i_ecg)
        else:         # normal
            total_N += 1
            if (np.array(pred_win)==1).sum() >= thresh_AF:
                FP_ECG += 1
                idx_FP = np.append(idx_FP,i_ecg)
                
    #TP_ECG_rate = TP_ECG / len(list_ECG) *100
    TP_ECG_rate = TP_ECG / total_P *100
    TP_ECG_rate_hist[thresh_AF] = TP_ECG_rate
    
    FP_ECG_rate = FP_ECG / total_N *100
    FP_ECG_rate_hist[thresh_AF] = FP_ECG_rate
    print(TP_ECG_rate,'   ',FP_ECG_rate)
#%%
#data = pickle.load(open('temp.p','rb'))
#TP_ECG_rate_hist = data['T']
#FP_ECG_rate_hist = data['F']
font =18
plt.figure(figsize=(18,100))
fig, axes = plt.subplots(1, 2, sharex=True,figsize=(2*15,2*5))

axes[0].grid()

#axes[0].plot(range(1,16),TP_ECG_rate_hist,'o-g')
axes[0].plot(TP_ECG_rate_hist[1:],'o-g')
#plt.plot(w.data[:,0],color = plt_color)
axes[0].set_title("TP Vs, AF_threshold", fontsize=font)
axes[0].set_xlabel('AF_Threshold', fontsize=font)
axes[0].set_yticks(np.arange(90,101,1))
axes[0].set_yticklabels(np.arange(90,101,1), fontsize=font)

axes[1].plot(FP_ECG_rate_hist[1:],'o-r')
#axes[1].plot(range(1,16),FP_ECG_rate_hist,'o-r')
axes[1].set_title("FP Vs, AF_threshold", fontsize=font)
axes[1].grid()
axes[1].set_xlabel('AF_Threshold', fontsize=font)
axes[1].set_yticks(np.arange(0,10,0.5))
axes[1].set_yticklabels(np.arange(0,10,0.5), fontsize=font)

#plt.figure()
#plt.plot(FP_ECG_rate_hist)
#plt.yticks = np.arange(0,100,10)
#plt.yticklabels = np.arange(0,100,10)

#flops, params = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False)

#    print("{:>40}  {:<8.2f}".format("Accuracy on all windows of test data:", acc))

#print("{:>40}  {:<8d}".format("Threshold for detecting AF:", thresh_AF))
#print("{:>40}  {:<8.3f}".format("TP rate:", TP_ECG_rate))
#print("{:>40}  {:<8.3f}".format("FP rate:", FP_ECG_rate))

#print('{:>40}  {:<8d}'.format('Number of parameters:', params))
#print('{:>40}  {:<8.0f}'.format('Computational complexity:', flops))

#return (TP_ECG_rate,idx_TP), (FP_ECG_rate,idx_FP), list_pred_win, elapsed