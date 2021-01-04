from default_modules import *
#import console
initite = 0
#%matplotlib inline    
# %% ================ loading data
data_dr = '/vol/hinkelstn/codes/'
result_dir = "/vol/hinkelstn/codes/flex_2c8,16_2f16_k8_s4_nobias_b512_raw_2K_last-512_nozsc_sub4_meannorm/"

if 'load_ECG' in locals() and initite == 0:
    print('''
        ==================      
          Using already extracted data
          ''')
    time.sleep(5)
else:    
#    load_ECG =  torch.load (data_dr+'raw_x_8K_sync_win2K.pt')
    load_ECG =  torch.load (data_dr+'raw_x_2K_nofilter_stable.pt')    
    


#save_name ="2d_6CN_3FC_no_BN_in_FC_long"
#save_name ="flex_2c8,16_2f16_k8_s4_sub4_b512_raw_2K_stable_cv1"
save_name = "flex_2c8,16_2f16_k8_s4_nobias_b512_raw_2K_last-512_nozsc_sub4_meannorm"+"_cv15"




raw_x = load_ECG['raw_x']
target = load_ECG['target']
data_tag = load_ECG['data_tag']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if type(target) != 'torch.Tensor':
    target = torch.tensor(load_ECG['target']).to(device)

loaded_vars = pickle.load(open(result_dir+"train_"+save_name+"_variables.p","rb"))

params = loaded_vars['params']
epoch = params.epoch
seed = params.seed
test_size = params.test_size
np.random.seed(seed)
t_range = params.t_range

if 'dataset_splits' not in locals():
    dataset_splits = create_datasets_win(raw_x, target, data_tag, test_size, seed=seed, t_range = t_range, device = device)

ecg_datasets = dataset_splits[0:3]
trn_idx, val_idx, tst_idx = dataset_splits[3:6]
trn_ds, val_ds, tst_ds = ecg_datasets

#batch_size = loaded_vars['params'].batch_size
batch_size = 1
trn_dl, val_dl, tst_dl = create_loaders(ecg_datasets, bs=batch_size, jobs = 0)
raw_feat = ecg_datasets[0][0][0].shape[0]
raw_size = ecg_datasets[0][0][0].shape[1]
num_classes = 2

#%%===============  loading a learned model

#save_name ="2d_6CN_3FC_no_BN_in_FC_long"

#model = pickle.load(open(result_dir+'train_'+save_name+'_best.pth', 'rb'))
model = torch.load(result_dir+'train_'+save_name+'_best.pth', lambda storage, loc: storage.cuda(device))


if prune ==1:
    prune_rate = 40
    w = model.FC[1].weight.data
    w1 =  w.view(-1,1)
    #            (v,ind) =torch.sort(w)
    (v,ind) = torch.sort(abs(w1), dim = 0)
    indc = np.ceil(prune_rate*len(ind)/100).astype(int)
    for i in ind[0:indc]: w1[i] = 0
    w1 = w1.reshape(w.shape)
    model.FC[1].weight.data = w1


#%%===============  report

#clear = lambda: os.system('clear') #on Windows System
#clear()


print ('''
==============================================================================
The summary of the 2-layer CNN network:
==============================================================================
       ''')
    
#raw_feat, raw_size = 2, 2048
    
print(model)



print ('''
       

       
==============================================================================
Table of the network parameters:         
==============================================================================
       ''')
    
model = model.to('cpu')    
summary(model, input_size=(raw_feat, raw_size), batch_size = batch_size, device = 'cpu')

#assert 1==2
print ('''
==============================================================================
Detail of the network's per layer computations and parameters:
==============================================================================
       ''')

flops, params = get_model_complexity_info(model, (raw_feat, raw_size), print_per_layer_stat=True, units = 'MMac')



#print ('''
#==============================================================================
#The network's accuracy:
#==============================================================================
#
#The network has the following accuracy:
#TP: 99.10 , FP: 3.88 AF_threshold = 3 (observing at least 3 AF signs to classify ECG as AF)
#TP: 98.77 , FP: 1.5 AF_threshold  = 7
#       ''')



# %%-------------- stats on weights

print ('''
==============================================================================
Statistical analysis of the weights:
==============================================================================        
       ''')    


def stat_analysis(params,title, weight = 'weights', color = 'g'):
    print ('Total '+weight+': %d' %(len(params)))
    n_zero = (params == 0).sum()
    print ('Number of zero '+weight+': %d' %(n_zero))
    
#    if 'ReLU' in title and weight is not 'weights':
    params = np.delete(params,np.where(params == 0))

    
    print ('Minimum of '+weight+': %3.7f' %(params.min()))
    print ('Maximum of '+weight+': %3.7f' %(params.max()))
    print ('Average value of '+weight+': %3.7f' %(params.mean()))
    print ('Variance of '+weight+': %3.7f' %(np.std(np.array(params))))
    
    plt.figure(figsize = (25,5))
    n, bins, patches = plt.hist(params, 30, facecolor = color , rwidth = 0.75)
    if max(bins) <1000 and min(bins) >-1000:
#        xbins = np.floor(bins)
#    else:
        xbins = np.floor(bins*10)/10
        plt.xticks(bins,xbins)
    plt.xlabel('Values')
    plt.ylabel('Counts')
    plt.title(title)
    plt.grid(True)
    plt.show()

    
params = [];
model = model.to('cpu')
for m in model.modules():
#    print(m)
#    print("========")
    if type(m) == nn.Conv2d:
#        print(m)
        params = np.append(params,np.array(m.weight.data.view(-1)))
#        np.append(params, m.weight.data)
    
    elif type(m) == nn.Linear:
#        print(m)        
        params = np.append(params,np.array(m.weight.data.view(-1)))
        
    elif type(m) == nn.BatchNorm2d:
#        print(m)        
        params = np.append(params,np.array(m.weight.data.view(-1)))
        



#print ('Minimum of absolute weights: %3.7f' %(np.abs(params).min()))

print('''
         
============= All weights of the network
          ''')
stat_analysis(params,'Histogram of all weights')

#%%  per layer weight

for i in range(len(model.raw)):
    params = []
    if type(model.raw[i]) == my_net_classes.SepConv1d_v5:
        for j in range(3):
            m = model.raw[i].layers[j]
    #        params = m.weight.data.view(-1)
            params = np.append(params,np.array(m.weight.data.view(-1)))
    #    if type(m)     == 
        print('''
    
              
    ============= Conv. Layer: %d '''
               %(i+1))
        stat_analysis(params,'Histogram of weights in Conv-layer: %d' %(i+1))

#------------------------------- FC layer
c = 0
for i in range(len(model.FC)):    
    
    m = model.FC[i].to('cpu')
    if type(m) == nn.Linear:
        c += 1
        params = np.array(m.weight.data.view(-1))
        print('''
              
    ============= Fully connected Layer: %d '''
               %(c))
        stat_analysis(params,'Histogram of weights in FC-layer: %d' %(c))

#c = 0
#for i in range(len(model.FC)):    
    
m = model.out[0].to('cpu')
#    if type(m) == nn.Linear:
#        c += 1
params = np.array(m.weight.data.view(-1))
print('''
          
============= Output Layer: '''
           )
stat_analysis(params,'Histogram of weights in output layer: ')        
        

# %%-------------- stats on feature maps
#%matplotlib inline    
print ('''


       

       

==============================================================================
Statistical analysis of the intermediate signals:
============================================================================== ''')    
    
model.eval()    
# Visualize feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

#i_batch,batch = next(enumerate(tst_dl))
#x_raw, y_target = [t.to(device) for t in batch]
x = trn_ds.tensors[0]
x = x.unsqueeze(2)

print('============= Feature-values, Input: ')

features = x.data.view(-1).to('cpu')        
stat_analysis(features ,"Histogram of feature-values, input layer", 'feature-values', color = 'b')           

model = model.to(device)
for i_layer in range(2):              #raw layers: 
#            print(i_layer)
    module = model.raw[i_layer]
    if type(model.raw[i_layer]) == nn.MaxPool2d:
        x = module(x)
        continue
    for (name,module) in model.raw[i_layer].layers._modules.items():
        x = module(x)
        if type(module) in (nn.Conv2d, nn.BatchNorm2d, nn.ReLU):
            features = x.data.view(-1).to('cpu')
#            if type(module) == nn.ReLU:
#                assert 1==2
#                features = np.delete(features,np.where(features == 0))
                
            
            module_n = type(module)

            print('''
                  
                  ''')
            print('============= Feature-values, layer %d ,'%(i_layer+1)+str(module_n)+' output: ')
        
            stat_analysis(features ,"Histogram of feature-values, layer %d , "
                          %(i_layer+1)+str(module_n)+' output', 'feature-values', color = 'b')           

for (name,module) in model.FC._modules.items():
    x = module(x)
    if type(module) in (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Linear):
        features = x.data.view(-1).to('cpu')
#        if type(module) == nn.ReLU:
#            features = np.delete(features,np.where(features == 0))
        
        module_n = type(module)

        print('''
              
              ''')
        print('============= Feature-values, FC layer,'+str(module_n)+' output: ')
    
        stat_analysis(features ,"Histogram of feature-values, layer %d , "
                      %(i_layer+1)+str(module_n)+' output', 'feature-values', color = 'b')

module  = model.out
x = module(x)
#if type(module) in (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Linear):
features = x.data.view(-1).to('cpu')
#if type(module) == nn.ReLU:
#    features = np.delete(features,np.where(features == 0))

#module_n = type(module)

print('''
      
      ''')
print('============= Feature-values, Output layer:')

stat_analysis(features ,"Histogram of feature-values, Output layer:", 'feature-values', color = 'b')           
