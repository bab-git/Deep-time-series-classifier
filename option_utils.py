#import gpustat
import glob
from torch import cuda, load

import my_net_classes as net

models = [    
    (net.Classifier_1d_3conv_2fc_4str_2sub,                    "3conv_2fc_4str_2sub"),  #sub-sample 2 -  1dconv - 3 conv - 2 FC   ready to quantize  - drop after relu + BN2d
    (net.Classifier_1d_4c_2fc_sub_qr,                    "1d_4c_2fc_sub2_qr"),  #sub-sample 2 -  1dconv - 4 conv - 2 FC   ready to quantize  - drop after relu + BN2d
    (net.Classifier_1d_6_conv_v2,                       "2d_6CN_3FC_no_BN_in_FC"), #"1dconv - 6 conv - 3 FC with dropout after relu + BN2d"    
#    (net.Classifier_4c_2f_k1_16_s1_8_sub,               "4c_2f_k16888_s8444_sub"),
#    (net.Classifier_4c_2f_s1_2_k1_4_p1_2_half_chan_sub, "4c_2f_k4888_s2444_p2000_halfchan_sub"),
#    (net.Classifier_1d_4_conv_2_fc_str_4_half_chan_sub, "4con_2fc_str_4_halfchan_sub"),
#    (net.Classifier_1d_4_conv_2_fc_str_4_k_4_sub,       "4con_2fc_str_4_k_4_sub"),
    (net.Classifier_1d_4_conv_2_fc_str_4_sub,           "4con_2fc_str_4_sub"),
    (net.Classifier_1d_4_conv_1_pool_2_fc_str_4,        "4con_1pool_2fc_str_4"),
#    (net.Classifier_1d_5_conv_2_fc_str_4_1out,          "5con_2fc_str_4_1out"),
#    (net.Classifier_1d_5_conv_2_fc_str_4_sigm_1out,     "5con_2fc_str_4_sigm_1out"),
#    (net.Classifier_1d_5_conv_2_fc_str_4_sigm_2out,     "5con_2fc_str_4_sigm_2out"),
#    (net.Classifier_1d_5_conv_2_fc_str_4,               "5con_2fc_str_4"),
# 	 (net.Classifier_1d_5_conv_1_pool_2_fc_k_4,          "5con_1pool_2fc_k_4"),
#    (net.Classifier_1d_5_conv_1_pool_2_fc_k1_4,         "5con_1pool_2fc_k1_4"),
#    (net.Classifier_1d_5_conv_1_pool_2_fc_1_out,        "1d_5con_1pool_2fc_1out"),
#    (net.Classifier_1d_5_conv_1_pool_2_fc,              "1d_5con_1pool_2fc"),
#    (net.Classifier_1d_5_conv_1_pool,                   "1d_5con_1pool"),
#    (net.Classifier_1d_6_conv_2_fc,                     "1d_6con_2fc"),
#    (net.Classifier_1d_6_conv_nodropbatch,              "1d_6con_nodropbatch"),
    (net.Classifier_1d_6_conv,                          "1d_6con")
]


#save_name ="2d_6CN_3FC_no_BN_in_FC_long"
#save_name ="2d_6CN_3FC_no_BN_in_FC"
#save_name = "test_full_6conv_long"
#save_name = "test_6conv_v2"
#save_name = "test_full_6conv"
#save_name = "1d_6_conv_qn"
#save_name = "1d_5conv_2FC_qn"
#save_name = "1d_3conv_2FC_v2_seed2"
#save_name = "1d_3conv_2FC_seed2"
#save_name = "1d_3conv_2FC_v2_2K_win"
#save_name = "1d_1_conv_1FC"
#save_name = "1d_3con_2FC_2K_win"
#save_name = "1d_6con_2K_win_2d"
#save_name = "1d_6con_2K_win_test_30"
#save_name = "1d_6con_b512_trim_2K_win"
#save_name = "1d_6con_b512_trim_2K_win_s11"
#save_name = "1d_6con_b512_trim_2K_win_s3"
#save_name = "1d_6con_b512_trim_2K_seed2"
#save_name = "1dconv_b512_t4K"
#save_name = "1dconv_b512_drop1B"
#save_name = "1dconv_b512_drop1"
#save_name = "batch_512_BN_B"
#save_name = "1dconv_b512_BNM_B"
#save_name = "1dconv_b512_BNA_B"
#save_name = "batch_512_BNA"
#save_name = "batch_512_BN"
#save_name = "batch_512_B"
#t_stamp = "_batch_512_11_29_17_03"


datasets = [
    ("raw_x_all_win2K_s1600_IDs.pt",    "trim_all_2K_win_1600_shift with saved IDs"),
    ("raw_x_all_win2K_s1600.pt",    "trim_all_2K_win_1600_shift"),
    ("raw_x_8K_sync_win1.5K.pt",    "trim_1.5K_win_400_shift"),
    ("raw_x_12K_sync_win1.5K.pt",   "trim_1.5K_win"),
    ("raw_x_10K_sync_win1K.pt",     "trim_1K_win"),
    ("raw_x_8K_sync_win2K.pt",      "trim_2K_win"),
    ("raw_x_12K_sync.pt",           "trim_12K"),
    ("raw_x_10K_sync.pt",           "trim_10K"),
    ("raw_x_8K_sync.pt",            "trim_8K"),
    ("raw_x_all.pt",                "trim_all")
]

def show_model_chooser(default = 0, override = None):
    if override:
        return models[override]
    print("\n".join(["{}: {}".format(i, name) for i, (_, name) in enumerate(models)]))
    idx = input("Choose model (default {}):".format(default))
    if idx.isdecimal():
        idx = int(idx)
        if idx in range(len(models)):
            return models[idx]
    return models[default]

def show_data_chooser(default = 5, override = None):
    if override:
        return datasets[override]
    print("\n".join(["{}: {}".format(i, file_name) for i, (file_name, _) in enumerate(datasets)]))
    idx = input("Choose dataset (default {}):".format(default))
    if idx.isdecimal():
        idx = int(idx)
        if idx in range(len(datasets)):
            return datasets[idx]
    return datasets[default]

def show_gpu_chooser(default = 0, override = None):
    if override:
        return override
    if not cuda.is_available():
        return "cpu"
    gpustat.new_query().print_formatted(no_color=True)
    idx = input("Choose GPU (default {}):".format(default))
    if idx.isdecimal():
        idx = int(idx)
        if idx in range(cuda.device_count()):
            return "cuda:{}".format(idx)
    return "cuda:{}".format(default)

def build_name(model_name, data_name, override = None):
    if override:
        return override
    save_name0 = "{}_{}".format(model_name, data_name)
    save_name = input('''Enter a save-file name for this trainig:
        default is {}   :'''.format(save_name0))
    save_name = save_name if save_name !='' else save_name0
    suffix = input('Enter any suffix for the save file (def: NONE):')
    return save_name+'_'+suffix

def find_save(model_name, data_name, override = None):
    if override:
        save_name = override
    else:
        save_name = glob.glob("*{}*{}*.p".format(model_name, data_name))
        if save_name != []:
            save_name.sort(key=len)
            save_name = save_name[0][6:-12] # File name without "train_" and "_variables.p"
        else:
            save_name = "NONE"
    save_name2 = input("Input saved model name (default {}) :".format(save_name))
    if save_name2 != '':
        save_name = save_name2
    return save_name

def load_partial_weights(model, file_name, device = "cpu"):
    if cuda.is_available():
        state_dict = load(file_name, map_location=lambda storage, loc: storage.cuda(device))
    else:
        state_dict = load(file_name, map_location=lambda storage, loc: storage)
    model_state = model.state_dict()
    total_num = 0
    for name, param in state_dict.items():
        if name not in model_state:
            continue
        if param.shape == model_state[name].shape:
            total_num += param.numel()
            model_state[name].copy_(param)
    print("Loaded {} weights from {}".format(total_num, file_name))