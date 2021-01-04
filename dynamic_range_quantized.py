model = quantized_model

print ('''
==============================================================================
The summary of the 2-layer CNN network:
==============================================================================
       ''')
    
#raw_feat, raw_size = 2, 2048
    
print(model)

# %%-------------- stats on weights

print ('''
==============================================================================
Statistical analysis of the weights:
==============================================================================        
       ''')    


def stat_analysis(params,title, weight = 'weights', color = 'g',zp=0):
    print ('Total '+weight+': %d' %(len(params)))

    if type(params) == torch.Tensor:
        params = params.numpy()

    n_zero = (params == zp).sum()
    print ('Number of zero '+weight+' (value = %d) : %d' %(zp,n_zero))
#    if 'ReLU' in title and weight is not 'weights':
    params = np.delete(params,np.where(params == zp))
    
    print ('Minimum of '+weight+': %3.7f' %(params.min()))
    print ('Maximum of '+weight+': %3.7f' %(params.max()))
    print ('Average value of '+weight+': %3.7f' %(np.mean(params)))
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
    if type(m) == nn.quantized.modules.conv.Conv2d:
#        print(m)
        params = np.append(params,np.array(m.weight().int_repr().data.view(-1)))
#        np.append(params, m.weight().int_repr().data)
    
    elif type(m) == nn.intrinsic.quantized.modules.linear_relu.LinearReLU:
#        print(m)        
        params = np.append(params,np.array(m.weight().int_repr().data.view(-1)))
        
    elif type(m) == nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d:
#        print(m)        
        params = np.append(params,np.array(m.weight().int_repr().data.view(-1)))
#        params = np.append(params,np.array(m.bias().int_repr().data.view(-1)))



#print ('Minimum of absolute weights: %3.7f' %(np.abs(params).min()))

print('''
         
============= All weights of the network
          ''')
stat_analysis(params,'Histogram of all weights')

#%%  per layer weight

for i in range(len(model.raw)):
    params = []
#    if type(model.raw[i]) == my_net_classes.SepConv1d_v5:
    for j in range(2):
        m = model.raw[i].layers[j]
#        params = m.weight().int_repr().data.view(-1)
        params = np.append(params,np.array(m.weight().int_repr().data.view(-1)))
#    if type(m)     == 
    print('''

              
============= Conv. Layer: %d '''
           %(i+1))
    stat_analysis(params,'Histogram of weights in Conv-layer: %d' %(i+1))

#------------------------------- FC layer
c = 0
for i in range(len(model.FC)):    
    
    m = model.FC[i].to('cpu')
    if type(m) == nn.intrinsic.quantized.modules.linear_relu.LinearReLU:
        c += 1
        params = np.array(m.weight().int_repr().data.view(-1))
        print('''
              
    ============= Fully connected Layer: %d '''
               %(c))
        stat_analysis(params,'Histogram of weights in FC-layer: %d' %(c))

#c = 0
#for i in range(len(model.FC)):    
    
m = model.out[0].to('cpu')
#    if type(m) == nn.Linear:
#        c += 1
params = np.array(m.weight().int_repr().data.view(-1))
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
x = x.to('cpu')
print('============= Feature-values, Input: ')
x = model.quant(x)
features = x.int_repr().data.view(-1).to('cpu')        
stat_analysis(features ,"Histogram of feature-values, input layer", 'feature-values', 
              color = 'b', zp = x.q_zero_point())

model = model.to(device)
for i_layer in range(2):              #raw layers: 
#            print(i_layer)
    module = model.raw[i_layer]
    if type(model.raw[i_layer]) == nn.MaxPool2d:
        x = module(x)
        continue
    for (name,module) in model.raw[i_layer].layers._modules.items():
        x = module(x)
        if type(module) in (nn.quantized.modules.conv.Conv2d, 
               nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d):
            features = x.int_repr().data.reshape(-1).to('cpu')
#            if type(module) == nn.ReLU:
#                assert 1==2
#                features = np.delete(features,np.where(features == 0))
                
            
            module_n = type(module)

            print('''
                  
                  ''')
            print('============= Feature-values, layer %d ,'%(i_layer+1)+str(module_n)+' output: ')
        
            stat_analysis(features ,"Histogram of feature-values, layer %d , "
                          %(i_layer+1)+str(module_n)+' output', 'feature-values', color = 'b', zp = module.zero_point)

for (name,module) in model.FC._modules.items():
    x = module(x)
    if type(module) == nn.intrinsic.quantized.modules.linear_relu.LinearReLU:
        features = x.int_repr().data.reshape(-1).to('cpu')
#        if type(module) == nn.ReLU:
#            features = np.delete(features,np.where(features == 0))
        
        module_n = type(module)

        print('''
              
              ''')
        print('============= Feature-values, FC layer,' + str(module_n)+': ')
    
        stat_analysis(features ,"Histogram of feature-values, FC layer, "
                      +str(module_n)+' output', 'feature-values', color = 'b', zp = module.zero_point)

module  = model.out[0]
x = module(x)
#if type(module) in (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Linear):
features = x.int_repr().data.reshape(-1).to('cpu')
#if type(module) == nn.ReLU:
#    features = np.delete(features,np.where(features == 0))

#module_n = type(module)

print('''
      
      ''')
print('============= Feature-values, Output layer:')

stat_analysis(features ,"Histogram of feature-values, Output layer:", 'feature-values', 
              color = 'b', zp = module.zero_point)
