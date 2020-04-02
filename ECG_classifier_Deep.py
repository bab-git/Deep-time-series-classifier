from default_modules import *
#%% =======================
seed = 1
seed2 = input ('Enter seed value for randomizing the splits (default = 1):')
if seed2 != '':
    seed = int(seed2)

np.random.seed(seed)

#==================== data IDs

#IDs = []
#main_path = '/vol/hinkelstn/data/FILTERED/atrial_fibrillation_8k/'    
#IDs.extend(os.listdir(main_path))
#IDs = os.listdir(main_path)
#main_path = '/vol/hinkelstn/data/FILTERED/sinus_rhythm_8k/'
#IDs.extend(os.listdir(main_path))
#
#target = np.ones(16000)
#target[0:8000]=0

#t_range = range(1000,1512)

#t_win = 2**11  #2048
t_win = 6016  #8000
#t_win = 2**11  #2048

#t_shift = 400
t_shift = None



dataset, data_name  = option_utils.show_data_chooser(default = 0)

print("{:>40}  {:<8s}".format("Loading dataset:", dataset))

load_ECG =  torch.load (data_dir+dataset)

#load_ECG =  torch.load ('raw_x_all.pt') 
#load_ECG =  torch.load ('raw_x_40k_50K.pt') 
#load_ECG =  torch.load ('raw_x_6K.pt') 
#load_ECG =  torch.load ('raw_x_8K_sync.pt')
#load_ECG =  torch.load ('raw_x_8K_sync_win2K.pt')
t_max = load_ECG['raw_x'][0].shape[1]
t_win_i = input("Enter the split out of {} (def: {}) : ".format(t_max,t_win)) 
t_win = int(t_win_i) if t_win_i != '' else int(t_win)
print("{:>40}  {:<8d}".format("Extracted window size :", t_win))

t_range = range(t_win)


slide = input("Sliding window? (def:no)")
slide = False if slide == '' else True
print("{:>40}  {:}".format("Sliding window mode:", slide))
#%%==================== test and train splits
print("{:>40}".format("creating datasets"))
    
#test_size = 0.25   
test_size = 0.3    #default

#cuda_num = input("enter cuda number to use: ")
cuda_num = 0   # export CUDA_VISIBLE_DEVICES=0

#device = torch.device('cuda:'+str(cuda_num) if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:'+str(cuda_num) if torch.cuda.is_available() and cuda_num != 'cpu' else 'cpu')
#if torch.cuda.is_available():
#    torch.cuda.device(cuda_num)
#device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

if t_shift:
    raw_x = load_ECG['raw_x']    
    length = raw_x.shape[2]
    n_win = np.floor(length / t_shift) - np.floor(t_win / t_shift)    
    raw_x_extend = torch.empty([raw_x.shape[0]*n_win.astype(int),raw_x.shape[1],t_win])
    target_extend = torch.empty(raw_x.shape[0]*n_win.astype(int))    
    target = load_ECG['target']

    print("Extracting temporal windows from ECG files..." )
                                      
    for i_data in range(raw_x.shape[0]):    
#        if i_data % 500 ==0:
#            print("data number: "+str(i_data))
        for i_win in range(int(n_win)):
                    
            i_ext = i_data*n_win+i_win
            
            raw_x_extend[int(i_ext),:,:] = raw_x[i_data,:,i_win*t_shift:i_win*t_shift+t_win] 
            target_extend[int(i_ext)] = target[i_data]                
    
    del raw_x, target
    raw_x = raw_x_extend.to(device)
    target = target_extend.to(device)
    
else:

    raw_x = load_ECG['raw_x']
#    raw_x = load_ECG['raw_x'].to(device)
    #raw_x.pin_memory = True
#    target = load_ECG['target']
    target = torch.tensor(load_ECG['target'])
    
    data_tag = load_ECG['data_tag']


dataset_splits = create_datasets_win(raw_x, target, data_tag, test_size, seed=seed, t_range = t_range, device = device)
ecg_datasets = dataset_splits[0:3]
trn_idx, val_idx, tst_idx = dataset_splits[3:6]


#ecg_datasets = create_datasets_file(raw_x, target, test_size, seed=seed, t_range = t_range, device = device)
#trn_ds, val_ds, tst_ds = create_datasets_file(raw_x, target, test_size, seed=seed, t_range = t_range)
trn_ds, val_ds, tst_ds = ecg_datasets

#ecg_datasets = create_datasets(IDs, target, test_size, seed=seed, t_range = t_range)

#print(dedent('''
#             Dataset shapes:
#             inputs: {}
#             target: {}'''.format((ecg_datasets[0][0][0].shape,len(IDs)),target.shape)))



print ('device is loaded to device:',device)
#%% ==================   Initialization              


batch_size = (2**3)*64   #default = 512
val_batch_size = batch_size  #default
#val_batch_size = 4 * batch_size 
#batch_size = 64
#batch_size = (2**5)*64
#batch_size = "full"
lr = 0.001
#n_epochs = 3000
n_epochs = 10000
#n_epochs = 30  # FP = 7.4%
#iterations_per_epoch = len(trn_dl)
num_classes = 2
best_acc = 0
patience, trials = 500, 0
base = 1
step = 2
loss_history = []
acc_history = []

trn_dl, val_dl, tst_dl = create_loaders(ecg_datasets, bs=batch_size, jobs = 0, bs_val = val_batch_size)

raw_feat = trn_ds[0][0].shape[0]
raw_size = trn_ds[0][0].shape[1]
trn_sz = len(trn_ds)


# custom weights initialization called on netG and netD
def weights_init(m):
#    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#        m.weight.data.normal_(0.0, 0.02)
#    elif classname.find('BatchNorm') != -1:
#        xavier(m.weight.data)
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.normal_(0.0, 0.02)
#        xavier(m.bias.data)
        
    elif isinstance(m, nn.BatchNorm2d):
#        xavier(m.weight.data)
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        m.running_mean.fill_(0)        

#---------------------- pruned model training
#save_name = "1d_4c_2fc_sub2_qr"
#load_model = 'prunned_1d_4c_2fc_sub2_qr_1fPi_20tpoch_iter_346'
#model = pickle.load(open(load_model+'.pth', 'rb'))
#print ('The loaded model is : ', load_model)
#model.apply(weights_init)
        
#---------------------- model from scratch
model_cls, model_name   = option_utils.show_model_chooser()
if model_name == '1d_flex_net':
    print('=============== Defining the network structure')
    convs = input ('convolutions separated by comma (e.g. 32,64) : ')
    convs = [int(i) for i in convs.split(",")]
    
    FCs = input ('FC layers separated by comma (e.g. 32,64) : ')
    FCs = [int(i) for i in FCs.split(",")]
    
    pool = input ('Input sub-sampling (e.g. 2): ')
#    pool = int(pool)  else 0
    
    rest_params = input ('Kernel size, strides, pad (def.: 8,4,2) : ')
    if rest_params == '':
        rest_params = [8,4,2]
    else:
        rest_params = [int(i) for i in rest_params.split(",")]
    kernels, strides, pads = rest_params
            
#    net = {'conv':convs, 'fc':FCs}
    net = {'conv':convs, 'fc':FCs, 'kernels': kernels, 'strides': strides, 'pads':pads}
    if pool not in (None, '', '0'):
        net['pre'] = 'pool' + pool
    
    print('Network summary:')
    print(net)
    
    model = model_cls(raw_feat, 2, t_win, net).to(device)

else:
    model = model_cls(raw_feat, num_classes, raw_size, batch_norm = True).to(device)

#model = my_net_classes.Classifier_1d_4c_2fc_sub_qr(raw_feat, num_classes, raw_size).to(device)
#model = my_net_classes.Classifier_1d_5_conv_v2(raw_feat, num_classes, raw_size).to(device)
#model = my_net_classes.Classifier_1d_1_conv_1FC(raw_feat, num_classes, raw_size).to(device)
#model = my_net_classes.Classifier_1d_3_conv_2FC_v2(raw_feat, num_classes, raw_size).to(device)
#model = my_net_classes.Classifier_1d_3_conv_2FC(raw_feat, num_classes, raw_size).to(device)
#model = my_net_classes.Classifier_1d_6_conv_v2(raw_feat, num_classes, raw_size).to(device)
#model = my_net_classes.Classifier_1d_6_conv(raw_feat, num_classes, raw_size).to(device)
#model = my_net_classes.Classifier_1d_6_conv(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
#model = my_net_classes.Classifier_1dconv(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
#model = my_net_classes.Classifier_1dconv_BN(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
#model = Classifier_1dconv(raw_feat, num_classes, raw_size/(2*4**3)).to(device)
#model = Classifier_1dconv(raw_feat, num_classes, raw_size).to(device)

#model2 = model.to('cpu')
#summary(model2, input_size=(raw_feat, raw_size), batch_size = batch_size, device = 'cpu')

#if torch.cuda.device_count() > 1:
#    print("Let's use", torch.cuda.device_count(), "GPUs!")    
#    model = nn.DataParallel(model,device_ids=[0,1,5]).cuda()



criterion = nn.CrossEntropyLoss (reduction = 'sum')

opt = optim.Adam(model.parameters(), lr=lr)

#print('Enter a save-file name for this trainig:')
print("chosen batch size: %d, test size: %2.2f" % (batch_size, test_size))
#save_name = input('''Enter a save-file name for this trainig: 
#    '''+'(default : '+model_name+'_trained)')

save_name = option_utils.build_name(model_name, data_name)
if model_name == '1d_flex_net':
#    suf0 = '_{}c_{}fc{}_pool{}_ksp{}{}{}'.format(len(convs),len(FCs),FCs[0],pool,kernels,strides,pads)
    suf0 = '_{}c{}_{}fc{}_pool{}'.format(len(convs),convs[0],len(FCs),FCs[0],pool)
else:
    suf0 = ''
    
#suf0 = ''
suffix = input('Enter any suffix for the save file (def:{}):'.format(suf0[1:]))
suffix = suf0 if suffix == '' else '_'+suffix
save_name += suffix
      
#if save_name =='':
#    save_name = load_model+'_trained'
print("Result will be saved file into : "+save_name)    
print('Start model training')
epoch = 0

#pickle.dump({'ecg_datasets':ecg_datasets},open("train_"+save_name+"_split.p","wb"))
#torch.save(ecg_datasets, 'train_'+save_name+'.pth')



#%%===============  Learning loop`
#millis = round(time.time())


#millis = round(time.monotonic() * 1000)
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
    for batch in val_dl:
        x_raw, y_batch = batch
#        x_raw, y_batch = [t.to(device) for t in batch]
#    x_raw, y_batch = [t.to(device) for t in val_ds.tensors]
        out = model(x_raw)
        preds = F.log_softmax(out, dim = 1).argmax(dim=1)
        total += y_batch.size(0)
        correct += (preds ==y_batch).sum().item()
        
    acc = correct / total * 100
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
        params = parameters(net, lr, epoch, patience, step, batch_size, t_range, seed, test_size)
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

#now = datetime.datetime.now()
#date_stamp = str(now.strftime("_%m_%d_%H_%M"))
#torch.save(model.state_dict(), 'best_ended_'+save_name+date_stamp+'.pth')
#params = parameters(lr, epoch, patience, step, batch_size, t_range)
#pickle.dump({'ecg_datasets':ecg_datasets},open("train_"+save_name+"_split.p","wb"))

print("Model is saved to: "+"train_"+save_name+'_best.pth')

#-----git push
#if os.path.isfile(load_file):
#repo = Repo(os.getcwd())
#repo.index.add(["variables_ended.p"])
#repo.index.add(["best_ended.pth"])
##repo.index.add(["variables.p"])
#repo.index.commit("Training finished on: "+str(datetime.datetime.now()))
#origin = repo.remotes.origin
#origin.push()

print('Done!')    


#%%===========================        
#def smooth(y, box_pts):
#    box = np.ones(box_pts)/box_pts
#    y_smooth = np.convolve(y, box, mode = 'same')
#    return y_smooth

#%%==========================  test result


thresh_AF = 5
win_size = 10

#TP_ECG_rate, FP_ECG_rate, list_pred_win, elapsed = evaluate(model, tst_dl, tst_idx, data_tag, thresh_AF = thresh_AF, device = device)
TP_ECG_rate, FP_ECG_rate, list_pred_win, elapsed = \
    evaluate(model, tst_dl, tst_idx, data_tag, thresh_AF = thresh_AF, 
             device = device, win_size = win_size, slide = slide)

assert 1==2
#%%==========================  visualize training curve
f, ax = plt.subplots(1,2, figsize=(12,4))    
ax[0].plot(loss_history, label = 'loss')
ax[0].set_title('Validation Loss History')
ax[0].set_xlabel('Epoch no.')
ax[0].set_ylabel('Loss')

ax[1].plot(smooth(acc_history, 5)[:-2], label='acc')
#ax[1].plot(acc_history, label='acc')
ax[1].set_title('Validation Accuracy History')
ax[1].set_xlabel('Epoch no.')
ax[1].set_ylabel('Accuracy');

#%%===============  checking internal values
drop=.5
batch_norm = True
model1 = nn.Sequential(
             _SepConv1d(2,  32, 8, 2, 3 )  #out: raw_size/str
#            SepConv1d(2,  32, 8, 2, 3, drop=drop, batch_norm = batch_norm),  #out: raw_size/str
#            SepConv1d(    32,  64, 8, 4, 2, drop=drop, batch_norm = batch_norm),
#            SepConv1d(    64, 128, 8, 4, 2, drop=drop, batch_norm = batch_norm),
#            SepConv1d(   128, 256, 8, 4, 2, drop=drop, batch_norm = batch_norm),
#            SepConv1d(   256, 512, 8, 4, 2, drop=drop, batch_norm = batch_norm),
#            SepConv1d(   512,1024, 8, 4, 2, batch_norm = batch_norm),
#            Flatten(),
#            nn.Linear(256, 64), nn.ReLU(inplace=True),
#            nn.Linear( 64, 64), nn.ReLU(inplace=True)
            ).to(device)
model_out = model1(x_raw)
#model_out = model1(x_raw[0,:,:])
x_raw.shape
model_out.shape


model2 = model.to('cpu')
summary(model2, input_size=(raw_feat, raw_size), batch_size = batch_size, device = 'cpu')