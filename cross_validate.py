import os, sys
import argparse
from copy import deepcopy
from collections import defaultdict
import torch
from torch import nn
from torch import optim
from torchsummary import summary
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from ptflops.flops_counter import get_model_complexity_info

import option_utils
from my_data_classes import create_datasets_cv
from my_net_classes import parameters
from training import train
from evaluation import evaluate

#%%

#def main(args):
os.chdir('/vol/hinkelstn/codes')

model, model_name   = option_utils.show_model_chooser()
dataset, data_name  = option_utils.show_data_chooser()
use_norm            = option_utils.show_normalization_chooser()
device              = option_utils.show_gpu_chooser(default=0)

n_splits = 5
n_repeats = 5
n_epochs = 10000
lr = 0.001
batch_size = 512
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
zero_mean = True

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

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

subsmpl = 4
if model_name == '1d_flex_net':
    net, model_name = option_utils.construct_flex_net()
    if 'pre' in net:  # Subsample before normalization; Not inside network
        subsmpl = net['pre']
        raw_x = raw_x[:, :, ::subsmpl]
        raw_size //= subsmpl
        del net['pre']
    
    model = model(raw_feat, num_classes, raw_size, net).to(device)
else:
    model = model(raw_feat, num_classes, raw_size, batch_norm=True).to(device)

# summary(model, (raw_feat, raw_size))
# exit()

init_state = deepcopy(model)
if model.out[0].out_features == 1:
    criterion = nn.BCEWithLogitsLoss(reduction = 'sum')
else:
    criterion = nn.CrossEntropyLoss(reduction = 'sum')
params = parameters(lr=lr, batch_size=batch_size, t_range=range(raw_size),
                    seed=seed, test_size=1 / n_splits, use_norm=use_norm,
                    zero_mean=zero_mean, sub=subsmpl,
                    cv_splits=n_splits, cv_repeats=n_repeats)

save_name = option_utils.build_name(model_name, data_name, batch_size, seed=seed, sub=subsmpl, use_norm=use_norm, zero_mean = zero_mean)

print(save_name)

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
    
    ecg_datasets = create_datasets_cv(raw_x, target, trn_idx, val_idx, tst_idx, use_norm, zero_mean, device)

    if orig_device:
        device = orig_device
    
    cv_save = "{}_cv{}".format(save_name, cv_idx)

    opt = optim.Adam(model.parameters(), lr=lr)

    model, tst_dl = train(model, ecg_datasets, opt, criterion, params, cv_save, 
                          save_name, batch_size=batch_size, n_epochs=n_epochs, 
                          loader_jobs=jobs, device=device, visualize=False)

    (tp[cv_idx],_), (fp[cv_idx],_), acc[cv_idx], _, elapsed[cv_idx] = evaluate(model, tst_dl, tst_idx, data_tag, device=device, slide=False, print_results=False)

    model = deepcopy(init_state)

print("Evaluating on test set")

flops, params = get_model_complexity_info(model, (raw_feat, raw_size), as_strings=False, print_per_layer_stat=False)

print("{:>40}  {:.2f} seconds".format("Mean elapsed test time:", elapsed.mean()))

print("{:>40}  {:.2f}".format("Min test accuracy:", acc.min()))
print("{:>40}  {:.2f}".format("Max test accuracy:", acc.max()))
print("{:>40}  {:.2f}".format("Mean test accuracy:", acc.mean()))
print("{:>40}  {:.2f}".format("Test accuracy standard deviation:", acc.std()))

print("{:>40}  {:.2f}".format("Min TP rate:", tp.min()))
print("{:>40}  {:.2f}".format("Max TP rate:", tp.max()))
print("{:>40}  {:.3f}".format("Mean TP rate:", tp.mean()))
print("{:>40}  {:.2f}".format("TP rate standard deviation:", tp.std()))

print("{:>40}  {:.2f}".format("Min FP rate:", fp.min()))
print("{:>40}  {:.2f}".format("Max FP rate:", fp.max()))
print("{:>40}  {:.3f}".format("Mean FP rate:", fp.mean()))
print("{:>40}  {:.2f}".format("FP rate standard deviation:", fp.std()))

print('{:>40}  {:d}'.format('Number of parameters:', params))
print('{:>40}  {:.0f}'.format('Computational complexity:', flops))