import time, os
import torch
import pickle

from my_data_classes import create_loaders

def train(model, ecg_datasets, opt, criterion, params, save_name, batch_size=512, val_batch_size=None, n_epochs=10000, loader_jobs=0, device="cpu", visualize=True):
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
        if epoch % base == 0:
            millis = (time.time())
            base_exp = True
        
        for batch in trn_dl:
            x_raw, y_batch = [t.to(device) for t in batch]
            opt.zero_grad()
            out = model(x_raw)
            # if single_out:
            #     y_batch = y_batch.float()
            #     out = out.squeeze()
            loss = criterion(out,y_batch)
            epoch_loss += loss.item()
            loss.backward()
            opt.step()
        
        epoch_loss /= trn_sz
        loss_history.append(epoch_loss)

        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for batch in val_dl:
                x_raw, y_batch = [t.to(device) for t in batch]
                out = model(x_raw)
                # if single_out:
                #     preds = torch.sigmoid(out).squeeze().round()
                # else:
                preds = out.argmax(dim=1)
                total += y_batch.size(0)
                correct += (preds ==y_batch).sum().item()
                
        acc = correct / total * 100
        acc_history.append(acc)

        if base_exp:
            millis2 = (time.time())
            print("model: "+save_name+" - Epoch %3d. Loss: %4f. Acc.: %2.2f epoch-time: %4.2f" % (epoch,epoch_loss,acc,(millis2-millis)))
            base *= step 
            base_exp = False
        
        if acc > best_acc:
            print("model: "+save_name+" - Epoch %d best model being saved with accuracy: %2.2f" % (epoch,best_acc))
            trials = 0
            best_acc = acc
            pickle.dump(model,open("train_"+save_name+'_best.pth','wb'))
            params.epoch = epoch
            params.patience = patience
            params.step = step
            pickle.dump({'params':params,'acc_history':acc_history, 'loss_history':loss_history},open("train_"+save_name+"_variables.p","wb"))
            
        else:
            trials += 1
            if trials >= patience:
                print('Early stopping on epoch %d' % (epoch))
                model = pickle.load(open("train_"+save_name+'_best.pth','rb'))
                model.opt = opt
                break
        epoch += 1
    
    print("Model is saved to: train_"+save_name+'_best.pth')

    #%%==========================  visualize training curve
    if visualize and os.environ.get('DISPLAY', None):
        f, ax = plt.subplots(1,2, figsize=(12,4))    
        ax[0].plot(loss_history, label = 'loss')
        ax[0].set_title('Validation Loss History')
        ax[0].set_xlabel('Epoch no.')
        ax[0].set_ylabel('Loss')

        ax[1].plot(smooth(acc_history, 5)[:-2], label='acc')
        #ax[1].plot(acc_history, label='acc')
        ax[1].set_title('Validation Accuracy History')
        ax[1].set_xlabel('Epoch no.')
        ax[1].set_ylabel('Accuracy')
        plt.show()

    model = pickle.load(open("train_"+save_name+'_best.pth','rb'))
    
    return model, tst_dl