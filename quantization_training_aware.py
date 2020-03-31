#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:18:34 2020

Training aware quantization


@author: bhossein
"""
from default_modules import *

#import torch
#import pickle
#%% ============== options
model_cls, model_name   = option_utils.show_model_chooser()
dataset, data_name  = option_utils.show_data_chooser(default = 0)
save_name           = option_utils.find_save(model_name, data_name, result_dir = result_dir)
if save_name in ['NONE','N']:
    save_name ="2c_2f_k88_s44_sub4_b512_raw_4K_stable"

print("{:>40}  {:<8s}".format("Selected experiment:", save_name))

#device              = option_utils.show_gpu_chooser(default=1)
cuda_num = 0   # export CUDA_VISIBLE_DEVICES=x
device = torch.device('cuda:'+str(cuda_num) if torch.cuda.is_available() and cuda_num != 'cpu' else 'cpu')
# %% ================ loading data
print("{:>40}  {:<8s}".format("Loading dataset:", dataset))

load_ECG = torch.load(data_dir+dataset)

slide = input("Sliding window? (def:yes)")
slide = True if slide == '' else False
print("{:>40}  {:}".format("Sliding window mode:", slide))
#%%===============  loading experiment's parameters and batches

print("{:>40}  {:<8s}".format("Loading model:", model_name))

loaded_vars = pickle.load(open(result_dir+"train_"+save_name+"_variables.p","rb"))
#loaded_file = pickle.load(open("variables"+t_stamp+".p","rb"))
#loaded_file = pickle.load(open("variables_ended"+t_stamp+".p","rb"))

params = loaded_vars['params']
epoch = params.epoch
print("{:>40}  {:<8d}".format("Epoch:", epoch))
seed = params.seed
print("{:>40}  {:<8d}".format("Seed:", seed))
test_size = params.test_size
np.random.seed(seed)
t_range = params.t_range

#cuda_num = input("cuda number:")
#cuda_num = 0   # export CUDA_VISIBLE_DEVICES=x

#device = torch.device('cuda:'+str(cuda_num) if torch.cuda.is_available() and cuda_num != 'cpu' else 'cpu')
#device = torch.device('cpu')

raw_x = load_ECG['raw_x']
#raw_x = load_ECG['raw_x'].to(device)
target = load_ECG['target']
data_tag = load_ECG['data_tag']
IDs = load_ECG['IDs'] if 'IDs' in load_ECG else []

if type(target) != 'torch.Tensor':
    target = torch.tensor(load_ECG['target']).to(device)


dataset_splits = create_datasets_win(raw_x, target, data_tag, test_size, seed=seed, t_range = t_range, device = device)
ecg_datasets = dataset_splits[0:3]
trn_idx, val_idx, tst_idx = dataset_splits[3:6]
#ecg_datasets = create_datasets_file(raw_x, target, test_size, seed=seed, t_range = t_range, device = device)


acc_history = loaded_vars['acc_history']
loss_history = loaded_vars['loss_history']
#ecg_datasets = loaded_split['ecg_datasets']
trn_ds, val_ds, tst_ds = ecg_datasets

batch_size = loaded_vars['params'].batch_size
trn_dl, val_dl, tst_dl = create_loaders(ecg_datasets, bs=batch_size, jobs = 0)
raw_feat = ecg_datasets[0][0][0].shape[0]
raw_size = ecg_datasets[0][0][0].shape[1]
num_classes = 2

#device = ecg_datasets[0].tensors[0].device
#device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

# %%
model_path = result_dir+'train_'+save_name+'_best.pth'

try:
    model = model_cls(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
    model0 = copy.deepcopy(model)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage.cuda(device)))
    else:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
except:    
    model = pickle.load(open(model_path, 'rb'))

thresh_AF = 5
win_size = 10

#TP_ECG_rate, FP_ECG_rate, list_pred_win, elapsed = evaluate(model, tst_dl, tst_idx, data_tag, thresh_AF = thresh_AF, device = device)
TP_ECG_rate, FP_ECG_rate, list_pred_win, elapsed = \
    evaluate(model, tst_dl, tst_idx, data_tag, thresh_AF = thresh_AF, 
             device = device, win_size = win_size, slide = slide)

summary(model.to('cpu'), input_size=(raw_feat, raw_size), batch_size = 1, device = 'cpu', Unit = 'KB')
#pickle.dump((TP_ECG_rate, FP_ECG_rate, list_pred_win, elapsed),open(save_name+"result.p","wb"))

assert 1==2

#%%
#device = ecg_datasets[0].tensors[0].device
#device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

#---------------------  Evaluation function
def evaluate(model, tst_dl, thresh_AF = 3, device = 'cpu'):
    model.to(device)
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
            list_x = list(range(i_batch*tst_dl.batch_size,min((i_batch+1)*tst_dl.batch_size,len(tst_ds))))
        #    x_raw, y_batch = [t.to(device) for t in batch]
        #    x_raw, y_batch = tst_ds.tensors
            #x_raw, y_batch = [t.to(device) for t in val_ds.tensors]
            out = model(x_raw)
            preds = F.log_softmax(out, dim = 1).argmax(dim=1)
        #    preds = F.log_softmax(out, dim = 1).argmax(dim=1).to('cpu')
            list_pred = np.append(list_pred,preds.tolist())
        #    list_pred = np.append(list_pred,preds.tolist())
            total += y_batch.size(0)
            correct += (preds ==y_batch).sum().item()    
        #    i_error = np.append(i_error,np.where(preds !=y_batch))
            i_error = np.append(i_error,[list_x[i] for i in np.where((preds !=y_batch).to('cpu'))[0]])
        #    TP += ((preds ==y_batch) & (1 ==y_batch)).sum().item()
        #    total_P += (1 ==y_batch).sum().item()
        #    FP += ((preds !=y_batch) & (0 ==y_batch)).sum().item()

    elapsed = time.time() - s
    print('''elapsed time (seconds): {0:.2f}'''.format(elapsed))
        
    acc = correct / total * 100
    #TP_rate = TP / total_P *100
    #FP_rate = FP / (total-total_P) *100
    
    print('Accuracy on all windows of test data:  %2.2f' %(acc))
    
    #TP_rate = TP / (1 ==y_batch).sum().item() *100
    #FP_rate = FP / (0 ==y_batch).sum().item() *100
    
    
    win_size = (data_tag==0).sum()
    # thresh_AF = win_size /2
    # thresh_AF = 3
    
    list_ECG = np.unique([data_tag[i] for i in tst_idx])
    #list_ECG = np.unique([data_tag[i] for i in tst_idx if target[i] == label])
    #len(list_error_ECG)/8000*100
    
    TP_ECG, FP_ECG , total_P, total_N = np.zeros(4)
    list_pred_win = 100*np.ones([len(list_ECG), win_size])
    for i_row, i_ecg in enumerate(list_ECG):
        list_win = np.where(data_tag==i_ecg)[0]
        pred_win = [list_pred[tst_idx.index(i)] for i in list_win]
    #    print(pred_win)
        list_pred_win[i_row,:] = pred_win    
                            
        if i_ecg >8000:   #AF
            total_P +=1
            if (np.array(pred_win)==1).sum() >= thresh_AF:
                TP_ECG += 1                    
        else:         # normal
            total_N +=1
            if (np.array(pred_win)==1).sum() >= thresh_AF:
                FP_ECG += 1
                
        
    #TP_ECG_rate = TP_ECG / len(list_ECG) *100
    TP_ECG_rate = TP_ECG / total_P *100
    FP_ECG_rate = FP_ECG / total_N *100
    
    
    print("Threshold for detecting AF: %d" % (thresh_AF))
    print("TP rate: %2.3f" % (TP_ECG_rate))
    print("FP rate: %2.3f" % (FP_ECG_rate))
    
    return TP_ECG_rate, FP_ECG_rate, list_pred_win, elapsed
#print('True positives on test data:  %2.2f' %(TP_rate))
#print('False positives on test data:  %2.2f' %(FP_rate))
#------------------------------------------   
# %%==================================================================== 
# ================================== Trining aware  Quantization
# ====================================================================     

def evaluation1(model_test,tst_dl, device = 'cpu', num_batch = len(tst_dl)):
    model_test.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        print("i_batch:", end =" ")
        for i_batch, batch in enumerate(tst_dl):
            if i_batch%10 == 0:
                print(i_batch, end =" ")
            x_raw, y_batch = [t.to(device) for t in batch]
            out = model_test(x_raw)
            preds = F.log_softmax(out, dim = 1).argmax(dim=1)    
            total += y_batch.size(0)
            correct += (preds ==y_batch).sum().item() 
            if i_batch >=num_batch:
                acc = correct / total * 100
                return acc
            
    acc = correct / total * 100
#    print("")
#    print(acc)    
    return acc
  
#model_qta = pickle.load(open('train_'+save_name+'_best.pth', 'rb'))
#device = torch.device('cuda:0')
device = torch.device('cuda:0' if torch.cuda.is_available() and cuda_num != 'cpu' else 'cpu')

model_qta = model.to(device)


#opt = torch.optim.SGD(model_qta.parameters(), lr = 0.0001) 
opt = torch.optim.Adam(model_qta.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss (reduction = 'sum')

ntrain_batches = 200000
n_epochs = 200


def train_one_epoch(model, criterion, opt, trn_dl, device, ntrain_batches):
    if ntrain_batches > len(trn_dl):
        ls = range(len(trn_dl))
    else:
        ls = random.sample(range(len(trn_dl)), ntrain_batches)
    
    model=model.to(device)
    print(next(model.parameters()).is_cuda)
    model.train()
    
    cnt = 0
    epoch_loss = 0
    
    for i, batch in enumerate(trn_dl):
#        break
        if i not in ls:
            continue
        print('.', end = '')
#        print('%d.'%(i), end = '')
        cnt += 1
#        x_raw, y_batch = batch
        x_raw, y_batch = [t.to(device) for t in batch]
        opt.zero_grad()
        out = model (x_raw)
        loss = criterion(out,y_batch)
        epoch_loss += loss.item()
        loss.backward()
        opt.step()
                
        if cnt >= ntrain_batches:
            print('Loss %3.3f' %(epoch_loss / cnt))
            return epoch_loss / cnt

    print('Full Loss %3.3f' %(epoch_loss / cnt))
    return epoch_loss / cnt

#model_qta = model_qta.to(device)

e_loss0 = train_one_epoch(model_qta, criterion, opt, trn_dl, device, ntrain_batches)

model_qta.eval()
model_qta.fuse_model()
model_qta.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model_qta, inplace=True)

print("=========  original floating point validation accuracy ===============")
acc = evaluation1(model,val_dl,device, 30)
print('%2.2f' %(acc))

print("")
print("=========  q-prepared floating point validation accuracy ===============")
acc_0 = evaluation1(model_qta,val_dl,device, 30)
print('%2.2f' %(acc_0))


#acc_0 = 0
for nepoch in range(n_epochs):
    e_loss = train_one_epoch(model_qta, criterion, opt, trn_dl, device, ntrain_batches)
    
    if nepoch > 3:
        # Freeze quantizer parameters
        model_qta.apply(torch.quantization.disable_observer)
    if nepoch > 2:
        # Freeze batch norm mean and variance estimates
        model_qta.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # Check the accuracy after each epoch
    model_qta.to('cpu')
    quantized_model = torch.quantization.convert(model_qta.eval(), inplace=False)
    quantized_model.eval()

#    acc = evaluation1(quantized_model,val_dl,'cpu', 30)
#    model_qta.to(device)
    acc = evaluation1(model_qta,val_dl,device , 30)
    
    print(', Epoch %d : accuracy = %2.2f'%(nepoch,acc))

    if acc > acc_0:
#        save_file = save_name+"_qta_full_train.p"
#        save_file = save_name+"_qta.p"
#        pickle.dump(model_qta,open(save_name+"qta_full_train.p",'wb'))
        pickle.dump(model_qta,open(save_file,'wb'))
        print ("file saved to :"+save_file)

#        torch.jit.save(torch.jit.script(quantized_model), 'quantized_model.pth')
        quantized_model_best = quantized_model
        model_qta_best = model_qta
        acc_0 = acc

    if e_loss < e_loss0:
        print("")
        print("original loss is reached")
        break
#    elif:
        
#quantized_model_best = model_best
#quantized_model = pickle.load(open("quantized_model.pth",'rb'))

thresh_AF = 7

print("=========  Q-trained floating point validation accuracy ===============")        
acc = evaluation1(model_qta_best,val_dl,'cpu', 30)
print('%2.2f' %(acc))

print("")
print("=========  Quantized-model validation accuracy ===============")
acc = evaluation1(quantized_model_best,val_dl,'cpu', 30)
print('%2.2f' %(acc))

print("")
print("=========  Q-trained floating point test result ===============")        
TP_ECG_rate_taq, FP_ECG_rate_taq,x,y = evaluate(model_qta_best,tst_dl, device = device, thresh_AF = thresh_AF)

print("=========  Qquantized-model test result ===============")        
TP_ECG_rate_taq, FP_ECG_rate_taq,x,y = evaluate(quantized_model_best,tst_dl, thresh_AF = thresh_AF)

assert 1==2
# %%======================= loading trained model_qta
save_file = save_name+"_qta_full_train.p"

model_qta_best = pickle.load(open(save_file, 'rb'))
model_qta_best.to('cpu')
quantized_model_best = torch.quantization.convert(model_qta_best.eval(), inplace=False)
quantized_model_best.eval()


# %%========================= saving to file
F = open("model_summary","w")

print("", file = F)
print("=================== Original Covolutional Neural Network Structure =========", file = F)
#print("", file = F)
print(model, file = F)
#print(summary(model_qta_best, input_size=(raw_feat, raw_size), batch_size = batch_size, device = 'cpu'), file = F)
print("", file = F)
print("=================== Quantized Covolutional Neural Network Structure =========", file = F)
#print("", file = F)
print(quantized_model_best, file = F)
print("", file = F)



F.close()

