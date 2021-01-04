import gpustat
from os.path import basename, split, join, normpath, abspath
import glob
from torch import cuda, load

import my_net_classes as net

models = [
    (net.Classifier_1d_flex_net,                        "1d_flex_net"),  #   flexibke network
    (net.Classifier_1c_1f_sub16_1out_k4,                "1c_1f_k4_sub16_1out"),
    (net.Classifier_1c_1f_sub16_1out,                   "1c_1f_sub16_1out"),
    (net.Classifier_1c_1f_sub8_1out,                    "1c_1f_sub8_1out"),
    (net.Classifier_1c_1f_sub8,                         "1c_1f_sub8"),
    (net.Classifier_1c_1f_sub4,                         "1c_1f_sub4"),
    (net.Classifier_1c_2f_sub4,                         "1c_2f_sub4"),
    (net.Classifier_2f_sub32,                           "2f_sub32_1out"),
    (net.Classifier_2f_sub16,                           "2f_sub16_1out"),
    (net.Classifier_2f_sub8,                            "2f_sub8_1out"),
    (net.Classifier_2f_sub4,                            "2f_sub4_1out"),
    (net.Classifier_1f_sub4,                            "1f_sub4"),
    (net.Classifier_ims_nn,                             "ims_nn"),
    (net.Classifier_1d_2_conv_2_fc_sub4,                "2c_2f_k88_s44_sub4"),
    (net.Classifier_4c_2f_k4_4_s4_2_sub,                "4c_2f_k8884_s4442_sub"),
    (net.Classifier_4c_2f_k1_16_s1_8_sub,               "4c_2f_k16888_s8444_sub"),
    (net.Classifier_4c_2f_s1_2_k1_4_p1_2_half_chan_sub, "4c_2f_k4888_s2444_p2000_halfchan_sub"),
    (net.Classifier_1d_4_conv_2_fc_str_4_half_chan_sub, "4con_2fc_str_4_halfchan_sub"),
    (net.Classifier_1d_4_conv_2_fc_str_4_k_4_sub,       "4con_2fc_str_4_k_4_sub"),
    (net.Classifier_1d_4_conv_2_fc_str_4_sub,           "4con_2fc_str_4_sub"),
    (net.Classifier_1d_4_conv_1_pool_2_fc_str_4,        "4con_1pool_2fc_str_4"),
    (net.Classifier_1d_5_conv_2_fc_str_4_1out,          "5con_2fc_str_4_1out"),
    (net.Classifier_1d_5_conv_2_fc_str_4_sigm_1out,     "5con_2fc_str_4_sigm_1out"),
    (net.Classifier_1d_5_conv_2_fc_str_4_sigm_2out,     "5con_2fc_str_4_sigm_2out"),
    (net.Classifier_1d_5_conv_2_fc_str_4,               "5con_2fc_str_4"),
	(net.Classifier_1d_5_conv_1_pool_2_fc_k_4,          "5con_1pool_2fc_k_4"),
    (net.Classifier_1d_5_conv_1_pool_2_fc_k1_4,         "5con_1pool_2fc_k1_4"),
    (net.Classifier_1d_5_conv_1_pool_2_fc_1_out,        "1d_5con_1pool_2fc_1out"),
    (net.Classifier_1d_5_conv_1_pool_2_fc,              "1d_5con_1pool_2fc"),
    (net.Classifier_1d_5_conv_1_pool,                   "1d_5con_1pool"),
    (net.Classifier_1d_6_conv_2_fc,                     "1d_6con_2fc"),
    (net.Classifier_1d_6_conv_nodropbatch,              "1d_6con_nodropbatch"),
    (net.Classifier_1d_6_conv,                          "1d_6con")
]

datasets = [
    ("raw_x_2K_nofilter_last-512_nozsc.pt", "raw_2K_last-512_nozsc"),
    ("raw_x_2K_nofilter_last-512.pt",       "raw_2K_last-512"),
    ("raw_x_2K_nofilter_last.pt",           "raw_2K_last"),
    ("raw_x_2K_nofilter_38Kstart.pt",       "raw_2K_38Kstart"),
    ("raw_x_2K_nofilter_middle.pt",         "raw_2K_mid"),
    ("raw_x_2K_nofilter_subsampled.pt",     "raw_2K_sub"),
    ("raw_x_8K_nofilter_stable_win2K.pt",   "raw_8K_stable_2K_win"),
    ("raw_x_2K_nofilter_stable.pt",         "raw_2K_stable"),
    ("raw_x_4K_nofilter_stable.pt",         "raw_4K_stable"),
    ("raw_x_6K_nofilter_stable.pt",         "raw_6K_stable"),
    ("raw_x_8K_nofilter_stable.pt",         "raw_8K_stable"),
    ("raw_x_all_win2K_s1600.pt",            "trim_all_2K_win_1600_shift"),
    ("raw_x_8K_sync_win1.5K.pt",            "trim_1.5K_win_400_shift"),
    ("raw_x_12K_sync_win1.5K.pt",           "trim_1.5K_win"),
    ("raw_x_10K_sync_win1K.pt",             "trim_1K_win"),
    ("raw_x_8K_sync_win2K.pt",              "trim_2K_win"),
    ("raw_x_12K_sync.pt",                   "trim_12K"),
    ("raw_x_10K_sync.pt",                   "trim_10K"),
    ("raw_x_8K_sync.pt",                    "trim_8K"),
    ("raw_x_all.pt",                        "trim_all")
]

def show_model_chooser(default = 0, override = None):
    if override != None:
        return models[override]
    print("\n".join(["{}: {}".format(i, name) for i, (_, name) in enumerate(models)]))
    idx = input("Choose model (default {}):".format(default))
    if idx.isdecimal():
        idx = int(idx)
        if idx in range(len(models)):
            return models[idx]
    return models[default]

def show_data_chooser(default = 0, override = None):
    if override != None:
        return datasets[override]
    print("\n".join(["{}: {}".format(i, file_name) for i, (file_name, _) in enumerate(datasets)]))
    idx = input("Choose dataset (default {}):".format(default))
    if idx.isdecimal():
        idx = int(idx)
        if idx in range(len(datasets)):
            return datasets[idx]
    return datasets[default]

def show_gpu_chooser(default = 0, override = None):
    if override != None:
        if override.isdecimal():
            idx = int(override)
            if cuda.is_available() and idx in range(cuda.device_count()):
                return "cuda:{}".format(idx)
        return override
    if not cuda.is_available():
        return "cpu"
    gpustat.new_query().print_formatted(no_color=True)
    idx = input("Choose GPU (default {}):".format(default))
    if idx == "cpu":
        return "cpu"
    if idx.isdecimal():
        idx = int(idx)
        if idx in range(cuda.device_count()):
            return "cuda:{}".format(idx)
    return "cuda:{}".format(default)

def show_normalization_chooser(default = False, override = None):
    if override != None:
        return override
    use_norm = default
    def_string = "Y/n" if default else "y/N"
    choice = input("Use normalization? {}:".format(def_string))
    if choice in ["yes", "Yes", "y", "Y"]:
        use_norm = True
    elif choice in ["no", "No", "n", "N"]:
        use_norm = False
    return use_norm

def build_name(model_name, data_name, batch_size, seed = 1, sub = None, use_norm = False, zero_mean = False, override = None):
    if override != None:
        return override
    if sub:
        data_name += "_sub{}".format(sub)
    if use_norm and zero_mean:
        data_name += "_meannorm"
    else:
        if use_norm:
            data_name += "_norm"
        if zero_mean:
            data_name += "_0mean"
    if seed != 1:
        data_name += "_seed{}".format(seed)
    return "{}_b{}_{}".format(model_name, batch_size, data_name)

def find_save(model_name, data_name, base_dir = "./", cv = False, override = None):
    cv_str = "_cv" if cv else ""
    if override != None:
        save_name = join(base_dir, override)
    elif model_name == "1d_flex_net":
        print("Base directory: {}".format(abspath(base_dir)))
        dirs = sorted(glob.glob(join(base_dir, "*flex_*/")))
        print("\n".join(["{}: {}".format(i, basename(normpath(name))) for i, name in enumerate(dirs)]))
        idx = input("Choose model dir or input dir path (default 3):")
        if idx.isdecimal():
            idx = int(idx)
            if idx in range(len(dirs)):
                save_dir = dirs[idx]
        elif idx != "":
            save_dir = idx
        else:
            save_dir = dirs[3]

        flex_list = sorted(set(glob.glob(join(save_dir, "*flex_*{}*.p".format(cv_str)))) - set(glob.glob(join(save_dir, "*cv[1-9]*"))))
        print("\n".join(["{}: {}".format(i, basename(name)[6:-12]) for i, name in enumerate(flex_list)]))
        idx = input("Choose model index or input save name (default 0):")
        if idx.isdecimal():
            idx = int(idx)
            if idx in range(len(flex_list)):
                return save_dir, basename(flex_list[idx])[6:-12]
        elif idx != "":
            return save_dir, basename(idx)
        else:
            return save_dir, basename(flex_list[0])[6:-12]
    else:
        print("Base directory: {}".format(abspath(base_dir)))
        dirs = sorted(glob.glob(join(base_dir, "*{}*{}*/".format(model_name, data_name))))
        print("\n".join(["{}: {}".format(i, basename(normpath(name))) for i, name in enumerate(dirs)]))
        idx = input("Choose model dir or input dir path (default 0):")
        if idx.isdecimal():
            idx = int(idx)
            if idx in range(len(dirs)):
                save_dir = dirs[idx]
        elif idx != "":
            save_dir = idx
        else:
            save_dir = dirs[0]
        
        model_list = sorted(set(glob.glob(join(save_dir, "*{}*.p".format(cv_str)))) - set(glob.glob(join(save_dir, "*cv[1-9]*"))))
        print("\n".join(["{}: {}".format(i, basename(name)[6:-12]) for i, name in enumerate(model_list)]))
        idx = input("Choose model index or input save name (default 0):")
        if idx.isdecimal():
            idx = int(idx)
            if idx in range(len(model_list)):
                return save_dir, basename(model_list[idx])[6:-12]
        elif idx != "":
            return save_dir, basename(idx)
        else:
            return save_dir, basename(model_list[0])[6:-12]

def construct_flex_net():
    print('=============== Defining the network structure')
    convstr = input('convolutions separated by comma (e.g. 32,64) : ')
    convs = [int(i) for i in convstr.split(",")]
    convstr = convs[0] if len(set(convs)) == 1 else convstr
    
    FCstr = input('FC layers separated by comma (e.g. 32,64) : ')
    FCs = [int(i) for i in FCstr.split(",")]
    FCstr = FCs[0] if len(set(FCs)) == 1 else FCstr
    
    pool = input('Input sub-sampling (e.g. 2): ')
    
    rest_params = input('Kernel size, strides, pad (def.: 8,4,2) : ')
    if rest_params == '':
        rest_params = [8,4,2]
    else:
        rest_params = [int(i) for i in rest_params.split(",")]
    kernels, strides, pads = rest_params
            
    net = {'conv': convs, 'fc': FCs, 'kernels': kernels, 'strides': strides, 'pads': pads}
    if pool not in (None, '', '0', '1'):
        net['pre'] = int(pool)

    save_name = "flex_{c}c{cd}_{f}f{fd}_k{k}_s{s}".format(c=len(convs), cd=convstr, f=len(FCs)+1, fd=FCstr, k=kernels, s=strides)
    
    print('Network summary: {}'.format(save_name))
    print(net)

    return net, save_name

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
