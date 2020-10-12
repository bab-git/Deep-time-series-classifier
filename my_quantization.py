#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:35:28 2020

@author: bhossein
"""
import torch
import copy
import random
from torch import nn
from torch import optim
from evaluation import evaluate

def evaluation1(model_test,tst_dl, device = 'cpu', num_batch = 0):
    if num_batch == 0:
        num_batch = len(tst_dl)
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
def train_one_epoch(model, criterion, opt, trn_dl, device, ntrain_batches):
    if ntrain_batches > len(trn_dl):
        ls = range(len(trn_dl))
    else:
        ls = random.sample(range(len(trn_dl)), ntrain_batches)
    
    model = model.to(device)
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
            print('not-complete epoch Loss %3.3f' %(epoch_loss / cnt))
            return epoch_loss / cnt

    print('Complete epoch loss %3.3f' %(epoch_loss / cnt))
    return epoch_loss / cnt  
#---------------------------------

def quant_aw_train(model, trn_dl, val_dl, val_idx, data_tag, device, TP_w = 2, n_epochs = 200, lr=0.0001):
    
    model_qta = copy.deepcopy(model)
    model_qta = model_qta.to(device)
    
    
    #opt = torch.optim.SGD(model_qta.parameters(), lr = 0.0001) 
    opt = torch.optim.Adam(model_qta.parameters(), lr=lr)
    
    criterion = nn.CrossEntropyLoss (reduction = 'sum')
    
    ntrain_batches = 200000
    #n_epochs = 20
    
    
    model_qta.eval()
    model_qta.fuse_model()
    model_qta.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    
    
    torch.backends.quantized.engine = 'fbgemm'
    torch.quantization.prepare_qat(model_qta, inplace=True)
    acc_0 = 0
    nepoch =1
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
        evaluate(quantized_model, val_dl, val_idx, data_tag,
                 device = 'cpu', slide = False, verbose = False)
        
    #    acc =  TP_w*np.sinh(TP_ECG_rate_taq[0]-90) + np.sinh(20-FP_ECG_rate_taq[0])
        acc = TP_w*TP_ECG_rate_taq[0] -FP_ECG_rate_taq[0]
        
        
    #    acc = evaluation1(quantized_model,val_dl,'cpu', 30)
        
        print(', Epoch %d : best_acc = %2.2f, accuracy = %2.2f'%(nepoch,acc_0, acc))
    
        if acc > acc_0:
    #        save_file = save_name+"_qta_full_train.p"
            save_file = save_name+"_qta_float"+"_"+str(TP_w)+"_"+str(lr)+"_"+prunned+".pth"
    #        save_file = save_name+"_qta.pth"
    #        save_file_Q = save_name+"_qta.pth"
            
            model_qta_best = copy.deepcopy(model_qta)
    
    #        pickle.dump(model_qta,open(save_name+"qta_full_train.p",'wb'))
#            pickle.dump(model_qta,open(result_dir+save_file,'wb'))
    #        checkpoint = {'model': model_cls,
    #                      'state_dict': model_qta.state_dict()}
    #        torch.save(quantized_model.state_dict(), result_dir+save_file)
    #        torch.save(model_qta.state_dict(), result_dir+save_file)
    #        torch.jit.save(torch.jit.script(model_qta), result_dir+save_file)
    #        pickle.dump(quantized_model,open(result_dir+save_file_Q,'wb'))
#            print ("file saved to :"+save_file)
    
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
    return model_qta_best