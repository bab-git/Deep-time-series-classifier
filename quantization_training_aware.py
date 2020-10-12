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
save_name           = option_utils.find_save(model_name, data_name, result_dir = result_dir, default = 2)
if save_name in ['NONE','N']:
    save_name ="1d_4c_2fc_sub2_qr"
    save_name = "2d_6CN_3FC_no_BN_in_FC_long"
    save_name = "flex_2c8,16_2f16_k8_s4_b512_raw_2K_last-512_nozsc_sub4_meannorm_cv12"

print("{:>40}  {:<8s}".format("Selected experiment:", save_name))

#device              = option_utils.show_gpu_chooser(default=1)
cuda_num = 0   # export CUDA_VISIBLE_DEVICES=x
device = torch.device('cuda:'+str(cuda_num) if torch.cuda.is_available() and cuda_num != 'cpu' else 'cpu')
# %% ================ loading data
print("{:>40}  {:<8s}".format("Loading dataset:", dataset))

load_ECG = torch.load(data_dir+dataset)

slide = input("Sliding window? (def:no)")
slide = False if slide == '' else True
print("{:>40}  {:}".format("Sliding window mode:", slide))

n_epochs = input("Quantization training epochs? (def:200)")
n_epochs = 200 if n_epochs == '' else int(n_epochs)
#print("{:>40}  {:}".format("Sliding window mode:", slide))
quantize_flag = input("Also performing quantization? (def. yes)")
quantize_flag = True if quantize_flag == '' else False

TP_w = input("Weight of TP in combined accuray measure: (def. 2) : ")
TP_w = 2 if TP_w == '' else float(TP_w)

prunned = input("Pruning tag? (def: prune18) : ")
prunned = "prune18" if prunned == '' else ""

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

zero_mean = True if hasattr(params, "zero_mean") and params.zero_mean else False


#cuda_num = input("cuda number:")
#cuda_num = 0   # export CUDA_VISIBLE_DEVICES=x

#device = torch.device('cuda:'+str(cuda_num) if torch.cuda.is_available() and cuda_num != 'cpu' else 'cpu')
#device = torch.device('cpu')

raw_x = load_ECG['raw_x']
#raw_x = load_ECG['raw_x'].to(device)
target = load_ECG['target']
data_tag = load_ECG['data_tag']
IDs = load_ECG['IDs'] if 'IDs' in load_ECG else []

if hasattr(params, "sub") and params.sub != None:
    raw_x = raw_x[:, :, ::params.sub]

if type(target) != 'torch.Tensor':
    target = torch.tensor(load_ECG['target']).to(device)


dataset_splits = create_datasets_win(raw_x, target, data_tag, test_size, seed=seed, t_range = t_range, 
                                     zero_mean = zero_mean, device = device )
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
if prunned != "":
    model_path = result_dir+'train_'+save_name+'_best_'+prunned+'.pth'
else:
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
    if type(model) == int:    
        model = torch.load(model_path, map_location=device)
    

thresh_AF = 5
win_size = 10

#TP_ECG_rate, FP_ECG_rate, list_pred_win, elapsed = evaluate(model, tst_dl, tst_idx, data_tag, thresh_AF = thresh_AF, device = device)
TP_ECG_rate, FP_ECG_rate, _, _, _ = \
    evaluate(model, tst_dl, tst_idx, data_tag, thresh_AF = thresh_AF, 
             device = device, win_size = win_size, slide = slide)

mem_size = summary(model.to('cpu'), input_size=(raw_feat, raw_size), batch_size = 1, device = 'cpu', Unit = 'KB', verbose = False)
print('Memory size: ' +str(mem_size))
#pickle.dump((TP_ECG_rate, FP_ECG_rate, list_pred_win, elapsed),open(save_name+"result.p","wb"))

if not quantize_flag:
    assert 1==2

#%%
#device = ecg_datasets[0].tensors[0].device
#device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
#------------------------------------------   
def evaluation1(model_test,tst_dl, device = 'cpu', num_batch = len(tst_dl)):
    model_test.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        print("i_batch:", end =" ")
        for i_batch, batch in enumerate(tst_dl):
#            print(i_batch)
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

#---------------------------------
def train_one_epoch(model, criterion, opt, trn_dl, device, ntrain_batches, verbose = True):
    if ntrain_batches > len(trn_dl):
        ls = range(len(trn_dl))
    else:
        ls = random.sample(range(len(trn_dl)), ntrain_batches)
    
    model = model.to(device)
    if verbose:
        print(next(model.parameters()).is_cuda)
    model.train()
    
    cnt = 0
    epoch_loss = 0
    
    for i, batch in enumerate(trn_dl):
#        break
        if i not in ls:
            continue
        if verbose:
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
            if verbose:
                print('not-complete epoch Loss %3.3f' %(epoch_loss / cnt))
            return epoch_loss / cnt

    if verbose:
        print('Complete epoch loss %3.3f' %(epoch_loss / cnt))
    return epoch_loss / cnt    
# %%==================================================================== 
# ================================== Trining aware  Quantization
# ====================================================================     

  
#model_qta = pickle.load(open('train_'+save_name+'_best.pth', 'rb'))
#device = torch.device('cuda:0')
device = torch.device('cuda:0' if torch.cuda.is_available() and cuda_num != 'cpu' else 'cpu')

model_qta = copy.deepcopy(model)
model_qta = model_qta.to(device)

lr=0.0001
#opt = torch.optim.SGD(model_qta.parameters(), lr = 0.0001) 
opt = torch.optim.Adam(model_qta.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss (reduction = 'sum')

ntrain_batches = 200000
#n_epochs = 20

#max_val, min_val = 10, -10
max_val, min_val = None, None

#model_qta = model_qta.to(device)

#e_loss0 = train_one_epoch(model_qta, criterion, opt, trn_dl, device, ntrain_batches)

model_qta.eval()
model_qta.fuse_model()
model_qta.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

#model_qta.qconfig = torch.quantization.QConfig(
#        activation = qt.MovingAverageMinMaxObserver(dtype=torch.qint8, reduce_range=True, qscheme = torch.per_channel_affine), 
#        weight=qt.default_observer.with_args(dtype=torch.qint8))


#model_qta.qconfig.activation.p.keywords['quant_max']=2**8-1
#model_qta.qconfig.weight.p.keywords['dtype'] = torch.qint32
#model_qta.qconfig.weight.p.keywords['quant_max'] = 2**7
#model_qta.qconfig.weight.p.keywords['quant_min'] = -2**7


torch.backends.quantized.engine = 'fbgemm'
torch.quantization.prepare_qat(model_qta, inplace=True)

print("=========  original floating point validation accuracy ===============")
acc = evaluation1(model,val_dl,device, 30)
print('%2.2f' %(acc))

print("")
#acc_0 = evaluation1(model_qta,trn_dl,device, 30)
print("=========  Intital q-prepared floating point validation accuracy ===============")
acc_0 = evaluation1(model_qta,val_dl,device, 30)
print('%2.2f' %(acc_0))


#model_mq.qconfig = my_qconfig 
#for m in model_qta.modules():
#    try:
#        m.observer.max_val = max_val if max_val else m.observer.max_val 
#        m.observer.min_val = min_val if min_val else m.observer.min_val
#    except:
#        continue

acc_0 = 0
nepoch =1
#e_loss0 = 10000
#%%
while nepoch < n_epochs:
    e_loss = train_one_epoch(model_qta, criterion, opt, trn_dl, device, ntrain_batches)
    
    if nepoch >= 300:
        # Freeze quantizer parameters
        model_qta.apply(torch.quantization.disable_observer)
    if nepoch > 200:
        # Freeze batch norm mean and variance estimates
        model_qta.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # Check the accuracy after each epoch
#    evaluation1(model_qta,val_dl,device, 30)
    
    model_qta.to('cpu')
    quantized_model = torch.quantization.convert(model_qta.eval(), inplace=False)
#    quantized_model.eval()

#    acc = evaluation1(quantized_model,val_dl,'cpu', 30)

    
    TP_ECG_rate_taq, FP_ECG_rate_taq, _, _,_ = \
    evaluate(quantized_model, val_dl, val_idx, data_tag, thresh_AF = thresh_AF, 
             device = 'cpu', win_size = win_size, slide = slide,
             verbose = False)
    
#    acc =  TP_w*np.sinh(TP_ECG_rate_taq[0]-90) + np.sinh(20-FP_ECG_rate_taq[0])
    acc = TP_w*TP_ECG_rate_taq[0] -FP_ECG_rate_taq[0]
    
    
#    acc = evaluation1(quantized_model,val_dl,'cpu', 30)
    
    print(', Epoch %d : best_acc = %2.2f, accuracy = %2.2f'%(nepoch,acc_0, acc))

    if acc > acc_0:
#        save_file = save_name+"_qta_full_train.p"
        save_file = save_name+"_qta_float"+"_"+str(TP_w)+"_"+str(lr)+"_"+prunned+".pth"
#        save_file = save_name+"_qta.pth"
#        save_file_Q = save_name+"_qta.pth"
        
#        model_qta_best = copy.deepcopy(model_qta)

#        pickle.dump(model_qta,open(save_name+"qta_full_train.p",'wb'))
        pickle.dump(model_qta,open(result_dir+save_file,'wb'))
#        checkpoint = {'model': model_cls,
#                      'state_dict': model_qta.state_dict()}
#        torch.save(quantized_model.state_dict(), result_dir+save_file)
#        torch.save(model_qta.state_dict(), result_dir+save_file)
#        torch.jit.save(torch.jit.script(model_qta), result_dir+save_file)
#        pickle.dump(quantized_model,open(result_dir+save_file_Q,'wb'))
        print ("file saved to :"+save_file)

#        torch.jit.save(torch.jit.script(quantized_model), 'quantized_model.pth')
#        quantized_model_best = quantized_model
#        model_qta_best = model_qta
        acc_0 = acc

#    if e_loss < e_loss0:
#        print("")
#        print("original loss is reached")
    nepoch += 1
#        break
#    elif:
#%%     
#save_file = "flex_2c8,16_2f16_k8_s4_b512_raw_2K_last-512_nozsc_sub4_meannorm_cv12_qta_float_1_0.0001_prune18.pth"
save_file = "flex_2c8,16_2f16_k8_s4_b512_raw_2K_last-512_nozsc_sub4_meannorm_cv12_qta_float_2_0.0001_prune18.pth"
#save_file = "flex_2c8,16_2f16_k8_s4_b512_raw_2K_last-512_nozsc_sub4_meannorm_cv12_qta_float_1_0.001_prune18.pth" # Bad!
#save_file = "flex_2c8,16_2f16_k8_s4_b512_raw_2K_last-512_nozsc_sub4_meannorm_cv12_qta_float_1.5_0.0001_prune18.pth" # Bad!
#save_file = "flex_2c8,16_2f16_k8_s4_b512_raw_2K_last-512_nozsc_sub4_meannorm_cv12_qta_float_1.5_prune18.pth"
#save_file = "flex_2c8,16_2f16_k8_s4_b512_raw_2K_last-512_nozsc_sub4_meannorm_cv12_qta_float_1.0_prune18.pth"
#save_file = "flex_2c8,16_2f16_k8_s4_b512_raw_2K_last-512_nozsc_sub4_meannorm_cv12_qta_float_2_prune18.pth"
#save_file = "flex_2c8,16_2f16_k8_s4_b512_raw_2K_last-512_nozsc_sub4_meannorm_cv12_qta_float_1.5_1e-05_prune18.pth"
#save_file = "flex_2c8,16_2f16_k8_s4_b512_raw_2K_last-512_nozsc_sub4_meannorm_cv12_qta_float_2_.pth"
#save_file = "flex_2c8,16_2f16_k8_s4_b512_raw_2K_last-512_nozsc_sub4_meannorm_cv12_qta_float_1.5_0.001_.pth"
#save_file = "flex_2c8,16_2f16_k8_s4_b512_raw_2K_last-512_nozsc_sub4_meannorm_cv12_qta_float_1.5_0.0001_.pth"
#save_file = "flex_2c8,16_2f16_k8_s4_b512_raw_2K_last-512_nozsc_sub4_meannorm_cv12_qta_float_2_0.001_.pth"
#save_file = "flex_2c8,16_2f16_k8_s4_b512_raw_2K_last-512_nozsc_sub4_meannorm_cv12_qta_float_1_0.0001_.pth"
#save_file = save_name+"_qta_float.pth"
#save_file = save_name+"_qta.pth"
print(save_file)
model_qta_best = pickle.load(open(result_dir+save_file,'rb'))

#stat = torch.load(result_dir+save_file)

#quantized_model.state_dict(stat)
#model_qta_best = copy.deepcopy(model_qta)
#model_qta_best.state_dict(stat)

#model_qta_best = model_qta_best.to(device)
#model_qta_best.eval()
#model_qta_best.fuse_model()
#model_qta_best.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
#torch.quantization.prepare_qat(model_qta_best, inplace=True)
#model_qta_best.load_state_dict(torch.load(result_dir+save_file, map_location=lambda storage, loc: storage))
    
#%     
model_qta_best.to('cpu')

#evaluation1(model_qta_best,val_dl,'cpu', 30)
manual_range = False

if manual_range:
#    model_qta_best.quant.observer.observer.max_val = 4
#    model_qta_best.quant.observer.observer.min_val = -4
    for i_layer in [0,1]:
        for (_,m) in model_qta_best.raw[i_layer].layers._modules.items():
            print(m)
#            print("----------")
            if type(m) == nn.qat.Conv2d and m.out_channels == m.in_channels:
               print("layer: "+str(i_layer))
               print(m.observer.observer.max_val)
               print(m.observer.observer.min_val)
               if i_layer == 0:
                   print("")                   
#                   m.observer.observer.max_val = 5
#                   m.observer.observer.min_val = -5
               else:
                   print("")
#                   print(m.observer.observer.max_val)
#                   print(m.observer.observer.min_val)                   
#                   m.observer.observer.max_val = 90
#                   m.observer.observer.min_val = -80
                   
           
            if type(m) == nn.intrinsic.qat.ConvReLU2d:
               if i_layer == 0:
                   print("")
#                   print(m.observer.observer.max_val)
#                   print(m.observer.observer.min_val)                   
#                   m.observer.observer.max_val = 10
#                   m.observer.observer.min_val = -50
                   
                
    #        try:
    #            m.observer.max_val = max_val if max_val else m.observer.max_val 
    #            m.observer.min_val = min_val if min_val else m.observer.min_val
    #        except:
    #            continue
        
quantized_model_best = torch.quantization.convert(model_qta_best.eval(), inplace=False)

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
#TP_ECG_rate_taq, FP_ECG_rate_taq,x,y = evaluate(model_qta_best,tst_dl, device = device, thresh_AF = thresh_AF)
TP_ECG_rate_taq, FP_ECG_rate_taq, _, _,_ = \
    evaluate(model_qta_best, tst_dl, tst_idx, data_tag, thresh_AF = thresh_AF, 
             device = device, win_size = win_size, slide = slide)
print("=========  Qquantized-model test result ===============")        
#TP_ECG_rate_taq, FP_ECG_rate_taq,x,y = evaluate(quantized_model_best,tst_dl, thresh_AF = thresh_AF)
TP_ECG_rate_taq, FP_ECG_rate_taq, _, _,_ = \
    evaluate(quantized_model_best, tst_dl, tst_idx, data_tag, thresh_AF = thresh_AF, 
             device = 'cpu', win_size = win_size, slide = slide,
             verbose = False)
print("{:>40}  {:<8.3f}".format("TP rate:", TP_ECG_rate_taq[0]))
print("{:>40}  {:<8.3f}".format("FP rate:", FP_ECG_rate_taq[0]))
    
    
assert 1==2
# %%======================= loading trained model_qta
save_file = save_name+"_qta_full_train.p"
 
model_qta_best = pickle.load(open(result_dir+save_file, 'rb'))
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


#%% Debug
i,batch = next(enumerate(trn_dl))
x_raw = batch[0].to('cpu')
model_qta = copy.deepcopy(model)
model_qta = model_qta.to(device)


#opt = torch.optim.SGD(model_qta.parameters(), lr = 0.0001) 
opt = torch.optim.Adam(model_qta.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss (reduction = 'sum')

#ntrain_batches = 20
#n_epochs = 20




#model_qta = model_qta.to(device)

#e_loss0 = train_one_epoch(model_qta, criterion, opt, trn_dl, device, 20)

model_qta.eval()
model_qta.fuse_model()
model_qta.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model_qta, inplace=True)
e_loss = train_one_epoch(model_qta, criterion, opt, trn_dl, device, 20)
model_qta.to('cpu')
quantized_model = torch.quantization.convert(model_qta.eval(), inplace=False)
quantized_model.eval()
quantized_model(x_raw).shape