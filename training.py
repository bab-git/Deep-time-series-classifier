import time, os
import torch
import pickle

from my_data_classes import create_loaders
from evaluation import evaluate

def train(model, ecg_datasets, opt, criterion, params, save_name,           
          val_idx, data_tag, thresh_AF, win_size, slide,
          result_dir = '',          
          batch_size=512, val_batch_size=None, n_epochs=10000, loader_jobs=0, 
          device="cpu", visualize=True, acc_eval = True):
    epoch = 0
    best_acc = 0
    patience, trials = 500, 0
    base = 1
    step = 2
    loss_history = []
    acc_history = []
    trn_sz = len(ecg_datasets[0])
    if type(device) != torch.device:
        device = torch.device(device)

    trn_dl, val_dl, tst_dl = create_loaders(ecg_datasets, bs=batch_size, bs_val=val_batch_size, jobs=loader_jobs)

    while epoch < n_epochs:
        
        model.train()
        epoch_loss = 0
        millis = (time.time())
        
    #    print('trainig....')
    #    for batch in trn_dl.dataset:
    #        break
        for i, batch in enumerate(trn_dl):
    #        break
            x_raw, y_batch = batch
    #        x_raw, y_batch = [t.to(device) for t in trn_ds.tensors]
            opt.zero_grad()
            out = model (x_raw)
            loss = criterion(out,y_batch)
            epoch_loss += loss.item()
            loss.backward()
            opt.step()
        
        epoch_loss /= trn_sz
        loss_history.append(epoch_loss)
    
    #    with torch.no_grad():
        model.eval()
        correct, total = 0, 0
        
    #    print('validation....')
        if acc_eval:    
            acc, temp = \
                evaluate(model, val_dl, val_idx, data_tag, thresh_AF = thresh_AF, 
                         device = device, win_size = win_size, slide = slide,
                         verbose = False, acc_eval = True)
        else:
            TP_ECG_rate_taq, FP_ECG_rate_taq, _, _ , _ = \
                evaluate(model, val_dl, val_idx, data_tag, 
                         device = device, slide = slide,
                         verbose = False)
        
            acc = 60 + 2*(TP_ECG_rate_taq[0]-90) + 20-FP_ECG_rate_taq[0]                
        
            
    #    for batch in val_dl:
    #        x_raw, y_batch = batch
    ##        x_raw, y_batch = [t.to(device) for t in batch]
    ##    x_raw, y_batch = [t.to(device) for t in val_ds.tensors]
    #        out = model(x_raw)
    #        preds = F.log_softmax(out, dim = 1).argmax(dim=1)
    #        total += y_batch.size(0)
    #        correct += (preds ==y_batch).sum().item()
    #        
    #    acc = correct / total * 100
            
        acc_history.append(acc)
    
        millis2 = (time.time())
    
        if epoch % base ==0:
    #       print('Epoch: {epoch:3d}. Loss: {epoch_loss:.4f}. Acc.: {acc:2.2%}')
           print("model: "+save_name+" - Epoch %3d. Loss: %4f. Acc.: %2.2f epoch-time: %4.2f" % (epoch,epoch_loss,acc,(millis2-millis)))
           base *= step 
           
        if acc > best_acc:
            print("model: "+save_name+" - Epoch %d best model being saved with accuracy: %2.2f" % (epoch,best_acc))
            trials = 0
            best_acc = acc
    #        torch.save(model, "train_"+save_name+'_best.pth')
    #        torch.save(model.state_dict(), "train_"+save_name+'_best.pth')
            pickle.dump(model,open(result_dir+"train_"+save_name+'_best.pth','wb'))
    #        pickle.dump({'epoch':epoch,'acc_history':acc_history},open("train_"+save_name+"variables.p","wb"))
#            params = parameters(net, lr, epoch, patience, step, batch_size, t_range, seed, test_size)
            pickle.dump({'params':params,'acc_history':acc_history, 'loss_history':loss_history},open(result_dir+"train_"+save_name+"_variables.p","wb"))
            
        else:
            trials += 1
            if trials >= patience:
                print('Early stopping on epoch %d' % (epoch))
    #            model.load_state_dict(torch.load("train_"+save_name+'_best.pth'))
                model = pickle.load(open(result_dir+"train_"+save_name+'_best.pth','rb'))
                model.opt = opt
                break
        epoch += 1
    
    print("Model is saved to: train_"+save_name+'_best.pth')

#    #%%==========================  visualize training curve
#    if visualize and os.environ.get('DISPLAY', None):
#        f, ax = plt.subplots(1,2, figsize=(12,4))    
#        ax[0].plot(loss_history, label = 'loss')
#        ax[0].set_title('Validation Loss History')
#        ax[0].set_xlabel('Epoch no.')
#        ax[0].set_ylabel('Loss')
#
#        ax[1].plot(smooth(acc_history, 5)[:-2], label='acc')
#        #ax[1].plot(acc_history, label='acc')
#        ax[1].set_title('Validation Accuracy History')
#        ax[1].set_xlabel('Epoch no.')
#        ax[1].set_ylabel('Accuracy')
#        plt.show()

    model = pickle.load(open(result_dir+"train_"+save_name+'_best.pth','rb'))
    
    return model, tst_dl