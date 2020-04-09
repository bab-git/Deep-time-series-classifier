import os, sys
import argparse
from copy import deepcopy
from collections import defaultdict
import torch
from torch import nn
from torch import optim
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from ptflops.flops_counter import get_model_complexity_info

import option_utils
from my_data_classes import create_datasets_cv
from my_net_classes import parameters
from training import train
from evaluation import evaluate

#%%
def main(args):
    os.chdir('/vol/hinkelstn/codes')

    model, model_name   = option_utils.show_model_chooser(override=args.model)
    dataset, data_name  = option_utils.show_data_chooser(override=args.data)
    use_norm            = option_utils.show_normalization_chooser(override=args.use_norm)
    device              = option_utils.show_gpu_chooser(default=1, override=args.device)

    n_splits = 5
    n_repeats = 2
    n_epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch_size
    seed = args.seed
    np.random.seed(seed)

    print("Using device: {}".format(device))

    device = torch.device(device)

    print("Loading dataset: {}".format(dataset))

    load_ECG =  torch.load(dataset)
    raw_x = load_ECG["raw_x"]
    target = torch.tensor(load_ECG['target'])
    data_tag = load_ECG['data_tag']
    raw_feat = raw_x.shape[1]
    raw_size = raw_x.shape[2]
    num_classes = len(np.unique(target))

    print("Loading model: {}".format(model_name))

    if model_name == '1d_flex_net':
        net, model_name = option_utils.construct_flex_net()
        
        model = model(raw_feat, num_classes, raw_size, net).to(device)
    else:
        model = model(raw_feat, num_classes, raw_size, batch_norm = True).to(device)
    
    init_state = deepcopy(model.state_dict())
    criterion = nn.CrossEntropyLoss(reduction = 'sum')
    params = parameters(lr=lr, batch_size=batch_size, t_range=range(raw_size),
                        seed=seed, test_size=1/n_splits, use_norm=use_norm, cv_splits=n_splits, cv_repeats=n_repeats)

    save_name = option_utils.build_name(model_name, data_name, batch_size, seed=seed, use_norm=use_norm, override=args.out)

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    tp = np.empty(n_splits*n_repeats)
    fp = np.empty(n_splits*n_repeats)
    acc = np.empty(n_splits*n_repeats)
    elapsed = np.empty(n_splits*n_repeats)
    for cv_idx, (trn_idx, tst_idx) in enumerate(rskf.split(raw_x, target)):
        trn_idx, val_idx = train_test_split(trn_idx, test_size=len(tst_idx), stratify=target[trn_idx], random_state=seed)

        jobs = 0
        orig_device = None
        if device.type == "cuda":
            gpu_mem = torch.cuda.get_device_properties(device).total_memory
            data_size = sys.getsizeof(raw_x.storage()) + sys.getsizeof(target.storage())
            if data_size >= gpu_mem*0.85: # 85% of total memory, just a guess
                jobs = os.cpu_count()
                orig_device = device
                device = torch.device("cpu")
        
        ecg_datasets = create_datasets_cv(raw_x, target, trn_idx, val_idx, tst_idx, use_norm, device)

        if orig_device:
            device = orig_device
        
        cv_save = "{}_cv{}".format(save_name, cv_idx)

        opt = optim.Adam(model.parameters(), lr=lr)

        model, tst_dl = train(model, ecg_datasets, opt, criterion, params, cv_save, batch_size=batch_size, n_epochs=n_epochs, loader_jobs=jobs, device=device, visualize=False)

        (tp[cv_idx],_), (fp[cv_idx],_), acc[cv_idx], _, elapsed[cv_idx] = evaluate(model, tst_dl, tst_idx, data_tag, device=device, slide=False, print_results=False)

        model.load_state_dict(init_state)

    print("Evaluating on test set")

    flops, params = get_model_complexity_info(model, (raw_feat, raw_size), as_strings=False, print_per_layer_stat=False)

    print("{:>40}  {:.2f} seconds".format("Mean elapsed test time:", elapsed.mean()))
    print("{:>40}  {:.2f}".format("Mean accuracy on test data:", acc.mean()))
    print("{:>40}  {:.3f}".format("Mean TP rate:", tp.mean()))
    print("{:>40}  {:.3f}".format("Mean FP rate:", fp.mean()))

    print('{:>40}  {:d}'.format('Number of parameters:', params))
    print('{:>40}  {:.0f}'.format('Computational complexity:', flops))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",         type=int,            default=None,  help="Index of model in option_utils model list (default: show chooser)")
    parser.add_argument("-d", "--data",          type=int,            default=None,  help="Index of data set in option_utils data list (default: show chooser)")
    parser.add_argument("-c", "--device",                             default=None,  help="Index of Cuda device or name of device to use (default: show chooser)")
    parser.add_argument("-n", "--use_norm",      action="store_true",                help="Normalize data to [-1,1] range")
    parser.add_argument("-k", "--splits",        type=int,            default=5,     help="Number of cross validation splits (default: 5)")
    parser.add_argument("-r", "--repeats",       type=int,            default=2,     help="Number of cross validation repeats (default: 2)")
    parser.add_argument("-e", "--epochs",        type=int,            default=10000, help="Maximum training epochs (default: 10000)")
    parser.add_argument("-l", "--learning_rate", type=float,          default=0.005, help="Learning rate used for training (default: 0.005)")
    parser.add_argument("-b", "--batch_size",    type=int,            default=512,   help="Batch size used for training (default: 512)")
    parser.add_argument("-s", "--seed",          type=int,            default=1,     help="Seed used for random operations (default: 1)")
    parser.add_argument("-o", "--out",                                default=None,  help="Name to save model as (default: autogenerate from parameters)")
    args = parser.parse_args()
    main(args)