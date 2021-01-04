#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:18:34 2020

Training aware quantization


@author: bhossein
"""
from default_modules import *



def extract_stable_part(data, win_size = 2048, stride = 512):
    assert(win_size % stride == 0)

    offset = len(data) % stride
    indices = np.arange(offset, len(data) - (stride - 1), stride)
    maxs = np.zeros((len(indices), 2))
    mins = np.zeros((len(indices), 2))
    over = np.zeros_like(indices)
    undr = np.zeros_like(indices)
    for i, idx in enumerate(indices):
        win = data[idx:idx + stride]
        maxs[i] = win.max(axis=0)
        mins[i] = win.min(axis=0)
        over[i] = (win < 100).sum()
        undr[i] = (win > 3900).sum()

    str_in_win = win_size // stride
    min_diff = np.inf
    best = 0
    for i, idx in enumerate(indices[:-(str_in_win - 1)]):
        chan_diff = maxs[i:i + str_in_win].max(axis=0) - mins[i:i + str_in_win].min(axis=0)
        if (chan_diff > 0).all():
            diff = chan_diff.sum() + over[i:i + str_in_win].sum() + undr[i:i + str_in_win].sum()
            if diff < min_diff:
                min_diff = diff
                best = idx
    
    return data[best:best + win_size]

def check_constant(channel_data):
    first = channel_data[0]
    for i in range(1, len(channel_data)):
        if channel_data[i] != first:
            return False
    return True


def tree_summary(estimator):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold


    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
        "the following tree structure:"
        % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %.3f else to "
                "node %s."
                % (node_depth[i] * "\t",
                    i,
                    children_left[i],
                    feature[i],
                    threshold[i],
                    children_right[i],
                    ))

#def main(args):
#========paths:
model_cls, model_name   = option_utils.show_model_chooser()
dataset, data_name  = option_utils.show_data_chooser(default = 0)
#save_name           = option_utils.find_save(model_name, data_name, override = 2)
#save_dir, save_name = option_utils.find_save(model_name, data_name, base_dir = result_dir)

#save_name           = option_utils.find_save(model_name, data_name, result_dir = result_dir, default = 2)


features = "./Features.xlsx"
#ecg_dir = '/vol/hinkelstn/data/'+'ecg_8k_wav'+'/sinus_rhythm_8k/'
ecg_dir = '/vol/hinkelstn/data/'+'ecg_8k_wav'+'/atrial_fibrillation_8k/'
qntz_dir = "/vol/hinkelstn/codes/flex_2c8,16_2f16_k8_s4_nobias_b512_raw_2K_last-512_nozsc_sub4_meannorm_bbk/"
cnn_save_dir = "/vol/hinkelstn/codes/flex_2c8,16_2f16_k8_s4_nobias_b512_raw_2K_last-512_nozsc_sub4_meannorm/"
cnn_save_name = "flex_2c8,16_2f16_k8_s4_nobias_b512_raw_2K_last-512_nozsc_sub4_meannorm"+"_cv15"
data_dir = '/vol/hinkelstn/codes'

#rf = "./rf9_nf_cv0_pickle.p"
#rf = "./2c_rf20_0mean_cv10.p"
#rf = "./2c_rf15_0mean_cv16.p"
#rf = "./2c_rf15_0mean_cv22.p"

#os.chdir('/home/bhossein/BMBF project/code_repo/piere')
load_ECG = torch.load(os.path.join(data_dir,dataset))


#========parameters:
verbose = False
window = 2048 
stride = 512
sub = 8
#rf_seed = 1
#rf_size = 10

xls = pd.read_excel(features)
#rf = pickle.load(open(rf, "rb"))
rf_model = pickle.load(open(rf, "rb"))
rf = rf_model['rf']
maxX = rf_model['maxX']
maxint = rf_model['maxint']
norm = rf_model['norm']
m_trn = norm[0]
v_trn = norm[1]


   
#%%    testing te trained RF    

pos = 0
neg = 0
tp = 0
fp = 0

ecg_count = 0
for ecg_file in glob.glob(os.path.join(ecg_dir, "*.wav")):  
    if "00019fb0-6b6a-4ccf-b818-b52221ec524c.wav" in ecg_file:
        break


#        break
#    ecg_count +=1

#    if ecg_count % 1000 ==0:
#        print(ecg_count)

#RF_sub = 4
#rf_size = 15
#var_fact = 4
#max_depth = 170
#ccp_alpha = 3.05e-4
zsc = False
unit = True
#save_res = False
#save_csv = True
norm = True
three_class = True
#quantize = True

#params = ims_loaded_vars["params"]
#seed = params.seed
#np.random.seed(seed)
#n_splits = params.cv_splits
#n_repeats = params.cv_repeats
#n_repeats = 5
loaded_vars = pickle.load(open(os.path.join(cnn_save_dir, "train_"+cnn_save_name+"_variables.p"),"rb"))


cnn_params = loaded_vars["params"]
use_norm = True if hasattr(cnn_params, "use_norm") and cnn_params.use_norm else False
#batch_size = cnn_params.batch_size    
zero_mean = True if hasattr(cnn_params, "zero_mean") and cnn_params.zero_mean else False


save_file = "train_"+cnn_save_name
model = torch.load(os.path.join(cnn_save_dir, "train_"+cnn_save_name+'_best.pth'), map_location=device)
model.to('cpu')

#%% Floating point CNN
writer = pd.ExcelWriter('CNN_FP_signals.xlsx', engine='xlsxwriter')
workbook  = writer.book
col = 0

ecg = wavio.read(ecg_file)
sheet = 'CNN Input'

df = pd.DataFrame(data =[''])
df.to_excel(writer, sheet, startcol=0, startrow=0, header = False, index = False)
worksheet = writer.sheets[sheet]

columns=['ECG raw']
df = pd.DataFrame(data = ecg.data)
worksheet.write_row(0,col, columns)
df.to_excel(writer, sheet, startcol=col, startrow=1, header = False, index = False)

       
ecg0 = ecg
ecg = ecg.data[len(ecg.data)-stride-window:len(ecg.data)-stride]
    
if check_constant(ecg[:,0]) or check_constant(ecg[:,1]):
    ecg = extract_stable_part(ecg0.data, window, stride)
    print(ecg_file)

#sheet = 'ECG-raw'
columns=['ECG segment']
df = pd.DataFrame(data = ecg)
col += 4
worksheet.write_row(0,col, columns)
df.to_excel(writer, sheet, startcol=col, startrow=1, header = False, index = False)


raw_x = torch.tensor(np.transpose(ecg), dtype = torch.float32)


if hasattr(cnn_params, "sub") and cnn_params.sub != None:
    print('subsampling for CNN:',cnn_params.sub)
    raw_x = raw_x[:, ::cnn_params.sub]

columns=['segment subsampled x4']
df = pd.DataFrame(data = np.transpose(raw_x.numpy()))
col += 4
worksheet.write_row(0,col, columns)
df.to_excel(writer, sheet, startcol=col, startrow=1, header = False, index = False)

    
min_x = raw_x.min(dim=1).values
max_x = raw_x.max(dim=1).values
raw_x -= raw_x.mean(dim=1, keepdim=True)
raw_x /= (max_x - min_x)[:,None]


columns=['mean-normalized']
df = pd.DataFrame(data = np.transpose(raw_x.numpy()))
col += 4
worksheet.write_row(0,col, columns)
df.to_excel(writer, sheet, startcol=col, startrow=1, header = False, index = False)

raw_xu = raw_x.unsqueeze(0)

#cv_save = "{}{}".format(cnn_save_name[:-1], 15)


x = raw_x.unsqueeze(1)
x = x.unsqueeze(0)
out = x
columns_list=['Conv.1 out', 'Conv.2 out', 'Batch norm out', 'ReLu']

for i_layer in range(2):
    sheet = 'Conv. layer'+str(i_layer+1)
    df = pd.DataFrame(data =[''])
    df.to_excel(writer, sheet, startcol=0, startrow=0, header = False, index = False)
    worksheet = writer.sheets[sheet]
    
    col = 0
    
    for i_layer2 in range(4):
        module = model.raw[i_layer].layers[i_layer2]
        out = module(out)
    
        columns = [columns_list[i_layer2]]
        df = pd.DataFrame(data = np.transpose(out.data.squeeze().numpy()))            
        worksheet.write_row(0,col, columns)
        df.to_excel(writer, sheet, startcol=col, startrow=1, header = False, index = False)
        col += out.shape[1]+2
        

sheet = 'FC layer'
columns_list=['Flatten out', 'FC out', 'ReLu']

df = pd.DataFrame(data =[''])
df.to_excel(writer, sheet, startcol=0, startrow=0, header = False, index = False)
worksheet = writer.sheets[sheet]

col = 0

for i_layer2 in range(3):
    module = model.FC[i_layer2]
    out = module(out)

    columns = [columns_list[i_layer2]]
    df = pd.DataFrame(data = np.transpose(out.data.squeeze().numpy()))            
    worksheet.write_row(0,col, columns)
    df.to_excel(writer, sheet, startcol=col, startrow=1, header = False, index = False)
    col += 3

module = model.out[0]
out = module(out)        
columns = ['output']
df = pd.DataFrame(data = np.transpose(out.data.squeeze().numpy()))            
worksheet.write_row(0,col, columns)
df.to_excel(writer, sheet, startcol=col, startrow=1, header = False, index = False)        






writer.save()        




#%% Quantized CNN
#model_qta_best  = pickle.load(open(os.path.join(qntz_dir, "train_"+cnn_save_name+"_qntzd"+'_best.pth'), 'rb'))
#model = torch.quantization.convert(model_qta_best.eval(), inplace=False)

writer = pd.ExcelWriter('CNN_FP_signals.xlsx', engine='xlsxwriter')
workbook  = writer.book

writerQ = pd.ExcelWriter('CNN_INT8_signals.xlsx', engine='xlsxwriter')
workbookQ  = writerQ.book

col = 0
colq = 0

ecg = wavio.read(ecg_file)
sheet = 'CNN Input'

df = pd.DataFrame(data =[''])
df.to_excel(writer, sheet, startcol=0, startrow=0, header = False, index = False)
worksheet = writer.sheets[sheet]

df.to_excel(writerQ, sheet, startcol=0, startrow=0, header = False, index = False)
worksheetQ = writerQ.sheets[sheet]

columns=['ECG raw']
df = pd.DataFrame(data = ecg.data)
worksheet.write_row(0,col, columns)
df.to_excel(writer, sheet, startcol=col, startrow=1, header = False, index = False)

       
ecg0 = ecg
ecg = ecg.data[len(ecg.data)-stride-window:len(ecg.data)-stride]
    
if check_constant(ecg[:,0]) or check_constant(ecg[:,1]):
    ecg = extract_stable_part(ecg0.data, window, stride)
    print(ecg_file)

#sheet = 'ECG-raw'
columns=['ECG segment']
df = pd.DataFrame(data = ecg)
col += 4
worksheet.write_row(0,col, columns)
df.to_excel(writer, sheet, startcol=col, startrow=1, header = False, index = False)


raw_x = torch.tensor(np.transpose(ecg), dtype = torch.float32)


if hasattr(cnn_params, "sub") and cnn_params.sub != None:
    print('subsampling for CNN:',cnn_params.sub)
    raw_x = raw_x[:, ::cnn_params.sub]

columns=['segment subsampled x4']
df = pd.DataFrame(data = np.transpose(raw_x.numpy()))
col += 4
worksheet.write_row(0,col, columns)
df.to_excel(writer, sheet, startcol=col, startrow=1, header = False, index = False)
worksheetQ.set_column(col, col+1, 16)

    
min_x = raw_x.min(dim=1).values
max_x = raw_x.max(dim=1).values
raw_x -= raw_x.mean(dim=1, keepdim=True)
raw_x /= (max_x - min_x)[:,None]


columns=['mean-normalized']
df = pd.DataFrame(data = np.transpose(raw_x.numpy()))
col += 4
worksheet.write_row(0,col, columns)
df.to_excel(writer, sheet, startcol=col, startrow=1, header = False, index = False)

#raw_xu = raw_x.unsqueeze(0)

#cv_save = "{}{}".format(cnn_save_name[:-1], 15)


x = raw_x.unsqueeze(1)
x = x.unsqueeze(0)
out = model.quant(x)

columns=['Q-Input']
df = pd.DataFrame(data = np.transpose(out.int_repr().data.squeeze().numpy()))
worksheetQ.write_row(0,colq, columns)
df.to_excel(writerQ, sheet, startcol=colq, startrow=1, header = False, index = False)
colq += out.shape[1]+2

df = pd.DataFrame(data = np.column_stack((out.q_scale(), out.q_zero_point())), columns = ['scale', 'zero point'])
df.to_excel(writerQ, sheet, startcol=colq, startrow=0, index = False)
worksheetQ.set_column(colq, colq+1, 16)
colq += 4

columns_list=['Conv.1 out', 'Conv+BN+Relu out']

for i_layer in range(2):
    sheet = 'Conv. layer'+str(i_layer+1)
    df = pd.DataFrame(data =[''])
    df.to_excel(writer, sheet, startcol=0, startrow=0, header = False, index = False)
    worksheet = writer.sheets[sheet]
    
    df.to_excel(writerQ, sheet, startcol=0, startrow=0, header = False, index = False)
    worksheetQ = writerQ.sheets[sheet]
    
    col = 0
    colq = 0
    
    for i_layer2 in range(2):
        module = model.raw[i_layer].layers[i_layer2]
        out = module(out)
    
        columns = [columns_list[i_layer2]]
        df = pd.DataFrame(data = np.transpose(out.dequantize().data.squeeze().numpy()))            
        worksheet.write_row(0,col, columns)
        df.to_excel(writer, sheet, startcol=col, startrow=1, header = False, index = False)
        col += out.shape[1]+2
                
        df = pd.DataFrame(data = np.transpose(out.int_repr().data.squeeze().numpy()))
        worksheetQ.write_row(0,colq, columns)
        df.to_excel(writerQ, sheet, startcol=colq, startrow=1, header = False, index = False)
        colq += out.shape[1]+1
        
        df = pd.DataFrame(data = np.column_stack((out.q_scale(), out.q_zero_point())), columns = ['scale', 'zero point'])
        df.to_excel(writerQ, sheet, startcol=colq, startrow=0, index = False)
        worksheetQ.set_column(colq, colq+1, 16)
        colq += 5

        

sheet = 'FC layer'
columns_list=['Flatten out', 'FC+ReLu out']

df = pd.DataFrame(data =[''])
df.to_excel(writer, sheet, startcol=0, startrow=0, header = False, index = False)
worksheet = writer.sheets[sheet]

df.to_excel(writerQ, sheet, startcol=0, startrow=0, header = False, index = False)
worksheetQ = writerQ.sheets[sheet]


col = 0
colq = 0

for i_layer2 in range(2):
    module = model.FC[i_layer2]
    out = module(out)

    columns = [columns_list[i_layer2]]
    df = pd.DataFrame(data = np.transpose(out.dequantize().data.squeeze().numpy()))            
    worksheet.write_row(0,col, columns)
    df.to_excel(writer, sheet, startcol=col, startrow=1, header = False, index = False)
    col += 3
    
    
    df = pd.DataFrame(data = np.transpose(out.int_repr().data.squeeze().numpy()))
    worksheetQ.write_row(0,colq, columns)
    df.to_excel(writerQ, sheet, startcol=colq, startrow=1, header = False, index = False)
    colq += 2
    
    df = pd.DataFrame(data = np.column_stack((out.q_scale(), out.q_zero_point())), columns = ['scale', 'zero point'])
    df.to_excel(writerQ, sheet, startcol=colq, startrow=0, index = False)
    worksheetQ.set_column(colq, colq+1, 16)
    colq += 5
    

module = model.out[0]
out = module(out)
columns = ['output']
df = pd.DataFrame(data = np.transpose(out.dequantize().data.squeeze().numpy()))            
worksheet.write_row(0,col, columns)
df.to_excel(writer, sheet, startcol=col, startrow=1, header = False, index = False)   

df = pd.DataFrame(data = np.transpose(out.int_repr().data.squeeze().numpy()))
worksheetQ.write_row(0,colq, columns)
df.to_excel(writerQ, sheet, startcol=colq, startrow=1, header = False, index = False)
colq += 2

df = pd.DataFrame(data = np.column_stack((out.q_scale(), out.q_zero_point())), columns = ['scale', 'zero point'])
df.to_excel(writerQ, sheet, startcol=colq, startrow=0, index = False)
worksheetQ.set_column(colq, colq+1, 16)
colq += 5     






writer.save()        
writerQ.save()

