import time
import numpy as np
import torch
from torch.nn import functional as F
from ptflops.flops_counter import get_model_complexity_info, print_model_with_flops

#---------------------  Evaluation function
def evaluate(model, tst_dl, tst_idx, data_tag, thresh_AF = 3, 
             device = 'cpu', acc_eval = False, win_size = None, 
             slide = True):
    model.to(device)
    input_shape = tuple(tst_dl.dataset.tensors[0].shape[1:3])
    s = time.time()
    model.eval()
    correct, total , total_P, = 0, 0, 0
#    TP , FP = 0,0
    
    batch = []
    i_error = []
    list_pred = []
    y_tst = []
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
            y_tst = np.append(y_tst,y_batch.cpu())            
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
    
    if acc_eval:
            return acc, list_pred
    
    TP_ECG, FP_ECG , total_P, total_N = np.zeros(4)
    idx_TP = []
    idx_FP = []
    
    if slide == False:
        TP_ECG = ((list_pred == y_tst) & (1 == y_tst)).sum().item()
        total_P = (1 ==y_tst).sum().item()
#        TP_ECG_rate = TP / total_P *100
        FP_ECG = ((list_pred !=y_tst) & (0 == y_tst)).sum().item() 
        total_N = total-total_P
#        FP_ECG_rate = FP / (total-total_P) *100
    else:
            
        if win_size == None:
            win_size = (data_tag==0).sum()
        # thresh_AF = win_size /2
        # thresh_AF = 3
        
        list_ECG = np.unique([data_tag[i] for i in tst_idx])
        #list_ECG = np.unique([data_tag[i] for i in tst_idx if target[i] == label])
        #len(list_error_ECG)/8000*100
        
    #    TP_ECG, FP_ECG , total_P, total_N = np.zeros(4)
    #    idx_TP = []
    #    idx_FP = []
        list_pred_win = 100*np.ones([len(list_ECG), win_size])
        for i_row, i_ecg in enumerate(list_ECG):
            list_win = np.where(data_tag==i_ecg)[0][:win_size]
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
    FP_ECG_rate = FP_ECG / total_N *100
    
    flops, params = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False)

#    print("{:>40}  {:<8.2f}".format("Accuracy on all windows of test data:", acc))
    
    print("{:>40}  {:d} / {:d}".format("Threshold for detecting AF:", thresh_AF, win_size))
    print("{:>40}  {:<8.3f}".format("TP rate:", TP_ECG_rate))
    print("{:>40}  {:<8.3f}".format("FP rate:", FP_ECG_rate))

    print('{:>40}  {:<8d}'.format('Number of parameters:', params))
    print('{:>40}  {:<8.0f}'.format('Computational complexity:', flops))
    
    return (TP_ECG_rate,idx_TP), (FP_ECG_rate,idx_FP), list_pred_win, elapsed
#print('True positives on test data:  %2.2f' %(TP_rate))
#print('False positives on test data:  %2.2f' %(FP_rate))
#------------------------------------------  