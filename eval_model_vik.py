import argparse
import os
import pickle
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack
import sklearn
import torch
from joblib import dump, load
from pandas.plotting import scatter_matrix
from ptflops.flops_counter import get_model_complexity_info
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer

import evaluation
import fraunhofer_test
import option_utils_vik
import pruning
from confidence_knn import predict_3_class, predict_3_class_simplified, predict_reliability, predict_reliability_simplified
from confidence_svm import train_svm
from my_data_classes import create_datasets_cv, create_datasets_win, create_loaders
from rf_tree_selection import get_rf_order, predict_rf_sorted


def main(args):
    os.chdir('/vol/hinkelstn/codes')

    model_name = ""
#    if not args.file:
        
    _, model_name   = option_utils_vik.show_model_chooser()
    dataset, data_name  = option_utils_vik.show_data_chooser()
    cnn_save_dir, cnn_save_name = option_utils_vik.find_save(model_name, data_name, cv=args.cv)
    device              = option_utils_vik.show_gpu_chooser(default=1, override=args.device)

    print("{:>40}  {:<8s}".format("Using device:", device))

    device = torch.device(device)

    print("{:>40}  {:<8s}".format("Loading dataset:", dataset))

    load_ECG = torch.load(dataset)

    print("{:>40}  {:<8s}".format("Loading CNN:", cnn_save_name))


    ims_save_name = args.file if args.file else "13_27_2_p40_softsign_cv0"
    ims_save_dir = "13_27_2_p40_softsign/"

    features = np.loadtxt(fname = "Features.txt")

    print("{:>40}  {:<8s}".format("Loading IMS-net:", ims_save_name))

    ims_loaded_vars = pickle.load(open(os.path.join(ims_save_dir, "train_"+ims_save_name+"_variables.p"),"rb"))
    cnn_loaded_vars = pickle.load(open(os.path.join(cnn_save_dir, "train_"+cnn_save_name+"_variables.p"),"rb"))
    if hasattr(ims_loaded_vars['params'], "cv_splits") and ims_loaded_vars['params'].cv_splits > 0:
        eval_cv(ims_save_name, cnn_save_name, features, load_ECG, ims_loaded_vars, cnn_loaded_vars, ims_save_dir, cnn_save_dir, device, args.thresh, args.knn, args.svm, args.norm, args.prune)
    else:
        eval_single(ims_save_name, device, load_ECG, ims_loaded_vars, args.thresh)

def eval_single(save_name, device, load_ECG, loaded_vars, thresh_AF):
    params = ims_loaded_vars['params']
    epoch = params.epoch
    print("{:>40}  {:<8d}".format("Epoch:", epoch))
    seed = params.seed
    test_size = params.test_size
    np.random.seed(seed)
    t_range = params.t_range
    use_norm = True if hasattr(params, "use_norm") and params.use_norm else False

    raw_x = load_ECG['raw_x']
    target = torch.tensor(load_ECG['target'])
    data_tag = load_ECG['data_tag']
    target_ECG = load_ECG['target_ecg'] if 'target_ecg' in load_ECG else [0]*8000+[1]*8000

    jobs = 0
    orig_device = None
    if device.type == "cuda":
        gpu_mem = torch.cuda.get_device_properties(device).total_memory
        data_size = sys.getsizeof(raw_x.storage()) + sys.getsizeof(target.storage())
        if data_size >= gpu_mem*0.85: # 85% of total memory, just a guess
            jobs = os.cpu_count()
            orig_device = device
            device = torch.device("cpu")

    dataset_splits = create_datasets_win(raw_x, target, data_tag, test_size, seed = seed, t_range = t_range, use_norm = use_norm, device = device)
    ecg_datasets = dataset_splits[0:3]
    trn_idx, val_idx, tst_idx = dataset_splits[3:6]


    acc_history = loaded_vars['acc_history']
    loss_history = loaded_vars['loss_history']
    trn_ds, val_ds, tst_ds = ecg_datasets

    batch_size = params.batch_size
    trn_dl, val_dl, tst_dl = create_loaders(ecg_datasets, bs = batch_size, jobs = jobs)
    raw_feat = raw_x.shape[1]
    raw_size = raw_x.shape[2]
    num_classes = len(np.unique(target))

    if orig_device:
        device = orig_device

    model = pickle.load(open("train_"+save_name+'_best.pth','rb'))

    TP_ECG_rate, FP_ECG_rate, acc, list_pred_win, elapsed  = evaluate(model, tst_dl, tst_idx, data_tag, thresh_AF = thresh_AF, device = device)

    if os.environ.get('DISPLAY', None):
        f, ax = plt.subplots(1,2, figsize=(12,4))    
        ax[0].plot(loss_history, label = 'loss')
        ax[0].set_title('Validation Loss History: '+save_name)
        ax[0].set_xlabel('Epoch no.')
        ax[0].set_ylabel('Loss')
        ax[0].grid()

        ax[1].plot(smooth(acc_history, 5)[:-2], label='acc')
        #ax[1].plot(acc_history, label='acc')
        ax[1].set_title('Validation Accuracy History: '+save_name)
        ax[1].set_xlabel('Epoch no.')
        ax[1].set_ylabel('Accuracy')
        ax[1].grid()
        plt.show()

def eval_cv(ims_save_name, cnn_save_name, features, load_ECG, ims_loaded_vars, cnn_loaded_vars, ims_save_dir, cnn_save_dir, device, conf_thresh, k, use_svm, norm, prune):
    params = ims_loaded_vars["params"]
    seed = params.seed
    np.random.seed(seed)
    n_splits = params.cv_splits
    n_repeats = params.cv_repeats

    cnn_params = cnn_loaded_vars["params"]
    use_norm = True if hasattr(cnn_params, "use_norm") and cnn_params.use_norm else False
    batch_size = cnn_params.batch_size

    print("{:>40}  {:d}".format("Cross validation splits:", n_splits))
    print("{:>40}  {:d}".format("Cross validation repeats:", n_repeats))

    ims_x = features[:,:13]
    ims_y = features[:,13:15]

    raw_x = load_ECG['raw_x']
    target = torch.tensor(load_ECG['target'])

    fft_x0 = scipy.fftpack.fft(raw_x[:,0].numpy())
    fft_x0 = np.abs(fft_x0[:,:raw_x.shape[2]//2])
    fft_x1 = scipy.fftpack.fft(raw_x[:,1].numpy())
    fft_x1 = np.abs(fft_x1[:,:raw_x.shape[2]//2])

    nf1 = np.mean(fft_x0, axis=-1)
    nf2 = np.mean(fft_x1, axis=-1)
    nf3 = np.max(raw_x[:,0].numpy(), axis=-1)# / 11
    nf4 = np.max(raw_x[:,1].numpy(), axis=-1)# / 11
    nf5 = np.min(raw_x[:,0].numpy(), axis=-1)# / 11
    nf6 = np.min(raw_x[:,1].numpy(), axis=-1)# / 15
    
    ims_x = np.append(ims_x, np.transpose([nf1, nf2, nf3, nf4, nf5, nf6]), axis=1)
    # ims_x = ims_x[:,[0, 2, 5, 6, 7, 8, 9, 11, 12, 15, 16, 17, 18]] # stable
    ims_x = ims_x[:,[0, 2, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17]] # mid, last - 512

    # plt.plot(fft_x[802])

    # plt.scatter(np.mean(fft_x0, axis=-1), np.mean(fft_x1, axis=-1), c=target)
    # plt.show()
    # exit()

    assert (ims_y[:,1] == target.numpy()).all()

    data_tag = load_ECG['data_tag']
    raw_feat = raw_x.shape[1]
    raw_size = raw_x.shape[2]
    num_classes = len(np.unique(target))

    # rel_y = predict_reliability(ims_x, ims_y[:,1], k)

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    ims_tp = np.zeros(n_splits*n_repeats)
    ims_fp = np.zeros(n_splits*n_repeats)
    ims_acc = np.zeros(n_splits*n_repeats)
    cnn_tp = np.zeros_like(ims_tp)
    cnn_fp = np.zeros_like(ims_fp)
    cnn_acc = np.zeros_like(ims_acc)
    nums_total = np.zeros_like(ims_tp)
    nums_pos = np.zeros_like(nums_total)
    nums_neg = np.zeros_like(nums_total)
    nums_cnn = np.zeros_like(nums_total)
    nums_pos_cnn = np.zeros_like(nums_total)
    nums_neg_cnn = np.zeros_like(nums_total)
    conf = np.ones_like(nums_cnn)
    three_class = True
    use_pca = False
    use_tree = True
    rf_size = 10
    rf_seed = 1
    rf_depth = np.empty(0)
    rf_params = np.empty(0)
    # ims_x = ims_x[:,12]
    # ims_x = ims_x.reshape(-1,1)

    # ims_x = np.log(ims_x + 1)
    # pca = PCA()
    # ims_x = pca.fit_transform(ims_x)
    # df = pd.DataFrame(ims_x)
    # scatter_matrix(df, diagonal="kde", alpha=0.2, c=ims_y[:,1])
    # plt.show()
    # # for feat in range(ims_x.shape[1]):
    # #     plt.subplot(7, 2, feat + 1)
    # #     plt.plot(ims_x[:,feat])
    # # plt.show()
    # exit()

    # selector = SelectFromModel(RandomForestClassifier(random_state=rf_seed, n_estimators=rf_size, ccp_alpha=1.1e-4), threshold=-np.inf, max_features=13)
    # feat_ctr = Counter()
    maxint = 12800#np.iinfo(np.uint32).max
    quant = KBinsDiscretizer(n_bins=maxint, encode="ordinal", strategy="kmeans")

    for cv_idx, (trn_idx, tst_idx) in enumerate(rskf.split(ims_x, ims_y[:,1])):
        # trn_idx, val_idx = train_test_split(trn_idx, test_size=len(tst_idx), stratify=target[trn_idx], random_state=seed)
        val_idx = None

        cv_save = "{}{}".format(ims_save_name[:-1], cv_idx)
        x_trn = ims_x[trn_idx]
        y_trn = ims_y[trn_idx]
        # rel_trn = predict_reliability(x_trn, y_trn[:,1], k)
        x_tst = ims_x[tst_idx]
        y_tst = ims_y[tst_idx]

        if use_pca:
            pca = PCA()
            pca.fit(x_trn)
            x_trn = pca.transform(x_trn)
            x_tst = pca.transform(x_tst)

        if norm:
            m_trn = x_trn.mean(axis=0)
            v_trn = x_trn.std(axis=0)

            x_trn = (x_trn - m_trn) / v_trn
            x_tst = (x_tst - m_trn) / v_trn

        # rel_trn = predict_reliability_simplified(x_trn, y_trn[:,1], x_tst, k)
        if three_class:
            if use_tree:
                rel_trn = predict_3_class(x_trn, y_trn[:,1], k)
            else:
                rel_trn = predict_3_class_simplified(x_trn, y_trn[:,1], x_tst, k)
        else:
            rel_trn = y_trn[:,1]

        nums_total[cv_idx] = len(tst_idx)
        nums_pos[cv_idx] = (y_tst[:,1] == 1).sum()
        nums_neg[cv_idx] = (y_tst[:,1] == 0).sum()

        if len(rel_trn) > 0:
            if use_svm:
                svm = train_svm(x_trn, rel_trn)
                rel_mask = svm.predict(x_tst).astype(bool)
            elif use_tree:
                # dt = tree.DecisionTreeClassifier(random_state=rf_seed, max_depth=7)
                selected_trn = x_trn
                selected_tst = x_tst
                # selected_trn = selector.fit_transform(x_trn, rel_trn)
                # selected_tst = selector.transform(x_tst)
                # feat_ctr.update(np.where(selector.get_support())[0])
                # selected_trn = quant.fit_transform(selected_trn)
                # selected_tst = quant.transform(selected_tst)
                # selected_trn = (selected_trn * 100000000).astype(np.int32)
                # selected_tst = (selected_tst * 100000000).astype(np.int32)
                # print(selected_tst)
                # exit()
                # print(np.where(selector.get_support()))
                # print("before: {}, after: {}".format(x_tst.shape, selected_tst.shape))
                dt = RandomForestClassifier(random_state=rf_seed, n_estimators=rf_size, ccp_alpha=4.0e-4, max_depth=30)
                dt.fit(selected_trn, rel_trn)
                # temp = dt.estimators_[0].tree_.threshold.astype(np.int32)
                # dt.estimators_[0].tree_.threshold[:] = temp
                internal = [[estimator.tree_.feature,
                             estimator.tree_.threshold,
                             estimator.tree_.children_left,
                             estimator.tree_.children_right,
                             np.argmax(estimator.tree_.value[estimator.tree_.feature == -2][:,0], axis=-1)]
                             for estimator in dt.estimators_]
                # tree_summary(dt.estimators_[0])
                # print(internal)
                # print(dt.estimators_[0].tree_.children_right)
                # print(np.argmax(dt.estimators_[0].tree_.value[dt.estimators_[0].tree_.feature == -2][:,0], axis=-1))
                # print(dt.estimators_[0].tree_.node_count)
                # print(help(sklearn.tree._tree.Tree))
                # exit()
                # dump(internal, open("000_rf/3c_rf{}_nf_norm_k2_cv{}.p".format(rf_size, cv_idx), "wb"))
                rf_params = np.append(rf_params, np.sum([estimator.tree_.node_count for estimator in dt.estimators_]))
                rf_depth = np.append(rf_depth, [estimator.tree_.max_depth for estimator in dt.estimators_])
                # rf_depth = dt.tree_.max_depth
                order = get_rf_order(dt, selected_trn, rel_trn, "pred")
                pred = predict_rf_sorted(dt, selected_tst, order)
                # pred = dt.predict(selected_tst)
                rel_mask = pred != 2
                rel_trn = pred
            else:
                # rel_mask = predict_reliability(x_trn, rel_trn, k-1, x_tst=x_tst)
                if three_class:
                    rel_mask = rel_trn != 2
                else:
                    knn = KNeighborsClassifier(n_neighbors=k)
                    knn.fit(x_trn, rel_trn)
                    rel_trn = knn.predict(x_tst)
                    rel_mask = np.ones_like(rel_trn).astype(bool)

            x_rel = x_tst[rel_mask]
            y_rel = y_tst[rel_mask]
            
            if len(x_rel) > 0:
                # ims_tp[cv_idx], ims_fp[cv_idx], ims_acc[cv_idx], below_thresh = fraunhofer_test.evaluate(x_rel, y_rel, cv_save, ims_save_dir, conf_thresh=conf_thresh, print_results=False)
                ims_tp[cv_idx] = ((rel_trn == y_tst[:,1]) & (rel_trn == 1)).sum().item()
                ims_fp[cv_idx] = ((rel_trn != y_tst[:,1]) & (rel_trn == 1)).sum().item()
                ims_acc[cv_idx] = (rel_trn == y_tst[:,1]).astype(int).sum().item()

            tst_idx = tst_idx[np.invert(rel_mask)]

        nums_cnn[cv_idx] = len(tst_idx)
        nums_pos_cnn[cv_idx] = (ims_y[tst_idx,1] == 1).sum()
        nums_neg_cnn[cv_idx] = (ims_y[tst_idx,1] == 0).sum()

        if len(tst_idx) == 0:
            continue

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

        trn_dl, val_dl, tst_dl = create_loaders(ecg_datasets, bs=batch_size, jobs=jobs)

        if orig_device:
            device = orig_device
        
        cv_save = "{}{}".format(cnn_save_name[:-1], cv_idx)

        model = torch.load(os.path.join(cnn_save_dir, "train_"+cv_save+'_best.pth'), map_location=device)

        if prune > 0:
            model = pruning.prune_fc(model, prune)

        (cnn_tp[cv_idx],_), (cnn_fp[cv_idx],_), cnn_acc[cv_idx], _, _ = evaluation.evaluate(model, tst_dl, tst_idx, data_tag, device=device, slide=False, print_results=False)
    # cnn_tp = nums_pos_cnn
    # cnn_fp = nums_neg_cnn
    # cnn_acc = nums_pos_cnn

    # flops, params = get_model_complexity_info(model, (raw_feat, raw_size), as_strings=False, print_per_layer_stat=False)

    # print("{:>40}  {:.2f} seconds".format("Mean elapsed test time:", elapsed.mean()))
    nums_ims = nums_total - nums_cnn
    nums_pos_ims = nums_pos - nums_pos_cnn
    nums_neg_ims = nums_neg - nums_neg_cnn

    # # IMS-only
    # acc = ims_acc / nums_ims
    # tp = ims_tp / nums_pos_ims
    # fp = ims_fp / nums_neg_ims

    # # CNN-only
    # acc = cnn_acc / nums_cnn
    # tp = cnn_tp / nums_pos_cnn
    # fp = cnn_fp / nums_neg_cnn

    # Full
    acc = (ims_acc + cnn_acc) / nums_total
    tp = (ims_tp + cnn_tp) / nums_pos
    fp = (ims_fp + cnn_fp) / nums_neg
    
    conf = conf - (nums_cnn / nums_total)

    # print("{:>40}  {:.2%}".format("Total data labeled as reliable:", rel_y.sum() / ims_y.shape[0]))
    # print("{:>40}  {}".format("Best Features:", sorted([x[0] for x in feat_ctr.most_common(13)])))

    if (nums_ims != nums_total).any():
        print("{:>40}  {:.2%}".format("Min IMS-net data:", conf.min()))
        print("{:>40}  {:.2%}".format("Max IMS-net data:", conf.max()))
        print("{:>40}  {:.2%}".format("Mean IMS-net data:", conf.mean()))
        print("{:>40}  {:.2%}".format("IMS-net data standard deviation:", conf.std()))

    print("{:>40}  {:.2%}".format("Min test accuracy:", acc.min()))
    print("{:>40}  {:.2%}".format("Max test accuracy:", acc.max()))
    print("{:>40}  {:.2%}".format("Mean test accuracy:", acc.mean()))
    print("{:>40}  {:.2%}".format("Test accuracy standard deviation:", acc.std()))

    print("{:>40}  {:.2%}".format("Min TP rate:", np.nanmin(tp)))
    print("{:>40}  {:.2%}".format("Max TP rate:", np.nanmax(tp)))
    print("{:>40}  {:.2%}".format("Mean TP rate:", np.nanmean(tp)))
    print("{:>40}  {:.2%}".format("TP rate standard deviation:", np.nanstd(tp)))

    print("{:>40}  {:.2%}".format("Min FP rate:", fp.min()))
    print("{:>40}  {:.2%}".format("Max FP rate:", fp.max()))
    print("{:>40}  {:.2%}".format("Mean FP rate:", fp.mean()))
    print("{:>40}  {:.2%}".format("FP rate standard deviation:", fp.std()))

    if use_tree:
        print("{:>40}  {:.0f}".format("Min RF params:", rf_params.min()))
        print("{:>40}  {:.0f}".format("Max RF params:", rf_params.max()))
        print("{:>40}  {:.2f}".format("Mean RF params:", rf_params.mean()))

        print("{:>40}  {:.0f}".format("Min RF max_depth:", rf_depth.min()))
        print("{:>40}  {:.0f}".format("Max RF max_depth:", rf_depth.max()))
        print("{:>40}  {:.2f}".format("Mean RF max_depth:", rf_depth.mean()))

    print("{:>40}  {}".format("Min TP > 90+std:", tp.min() > 0.9 + tp.std()))
    print("{:>40}  {}".format("Mean TP > 90+4*std:", tp.mean() > 0.9 + (4 * tp.std())))
    print("{:>40}  {}".format("Max FP < 20-std:", fp.max() < 0.2 - tp.std()))
    print("{:>40}  {}".format("Mean FP < 20-4*std:", fp.mean() < 0.2 - (4 * fp.std())))

    # print('{:>40}  {:d}'.format('Number of parameters:', params))
    # print('{:>40}  {:.0f}'.format('Computational complexity:', flops))

    
    df = pd.DataFrame({"Total-Acc": acc * 100, "Total-TP": tp * 100, "Total-FP": fp * 100,
                       "IMS-Acc": ims_acc / nums_ims * 100, "IMS-TP": ims_tp / nums_pos_ims * 100, "IMS-FP": ims_fp / nums_neg_ims * 100,
                       "CNN-Acc": cnn_acc / nums_cnn * 100, "CNN-TP": cnn_tp / nums_pos_cnn * 100, "CNN-FP": cnn_fp / nums_neg_cnn * 100})
    # df.to_csv("{}_+_{}_k{}_r{}_{}nn_{}.csv".format(os.path.join(cnn_save_dir, ims_save_name[:-4]), cnn_save_name[:-4], n_splits, n_repeats, k, "svm" if use_svm else "rf8_3c"),
    #           index=False, float_format="%.2f")

    # if os.environ.get('DISPLAY', None):
    #     plt.boxplot([acc, tp, fp], labels=["Accuracy", "True Positives", "False Positives"], showmeans=True)
    #     plt.title("{}, {} splits, {} repeats".format(ims_save_name[:-4], n_splits, n_repeats))
    #     plt.show()
        # plt.figure(figsize=[15,10], dpi=150)
        # tree.plot_tree(dt, filled=True, class_names=["Normal", "AF", "Unreliable"])
        # plt.show()

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
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                "node %s."
                % (node_depth[i] * "\t",
                    i,
                    children_left[i],
                    feature[i],
                    threshold[i],
                    children_right[i],
                    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",         type=int,            default=None,  help="Index of model in option_utils model list (default: show chooser)")
    parser.add_argument("-d", "--data",          type=int,            default=None,  help="Index of data set in option_utils data list (default: show chooser)")
    parser.add_argument("-c", "--device",                             default=None,  help="Index of Cuda device or name of device to use (default: show chooser)")
    parser.add_argument("-p", "--prune",         type=float,          default=0,     help="Relative or absolute amount of weights to prune (default: 0)")
    parser.add_argument("-t", "--thresh",        type=float,          default=0.5,   help="Threshold for number of windows to be classified as AF. Total amount or relative to ECG window number. (default: 3)")
    parser.add_argument("-k", "--knn",           type=int,            default=2,     help="Number of considered neighbors for reliability labeling, including the sample itself (default: 2)")
    parser.add_argument("--cv",                  action="store_true",                help="Try to find cross validation saves when autodetecting")
    parser.add_argument("--svm",                 action="store_true",                help="Use SVM for reliability prediction (default: use kNN)")
    parser.add_argument("-n", "--norm",          action="store_true",                help="Normalize features")
    parser.add_argument("-f", "--file",                               default=None,  help="Name of saved model (default: autodetect)")
    args = parser.parse_args()
    main(args)
