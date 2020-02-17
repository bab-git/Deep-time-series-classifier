import argparse
import numpy as np
import torch
import wavio
import os
from scipy import stats
from my_data_classes import wave_harsh_peaks_all


def main(args):
    os.chdir('/vol/hinkelstn/codes')

    t_win   = args.window_size
#    print(t_win)
    t_shift = args.window_shift
    t_base  = args.peak_window
    out     = args.out

    target, raw_x, IDs, out_IDs = read_data(t_win, t_shift, t_base, out)
    extract_windows(t_win, t_shift, target, raw_x, IDs, out_IDs, out)

def read_data(t_length, t_shift, t_base, save_file):
    path_data = '/vol/hinkelstn/data/FILTERED/sinus_rhythm_8k/'
    path_data = np.append(path_data,'/vol/hinkelstn/data/FILTERED/atrial_fibrillation_8k/')
    main_path = path_data[0]
    IDs = os.listdir(main_path)
    main_path = path_data[1]
    IDs.extend(os.listdir(main_path))    
    idx = np.argsort(IDs)
    IDs.sort()

    target = np.ones(16000)
    target[0:8000]=0      # 0 : normal 1:AF
    target = [int(target[i]) for i in idx]

    min_length = ((t_length // t_shift) + 1) * t_shift # First multiple of t_shift larger than t_length

    out_IDs = np.empty((0,2))
    
    raw_x = torch.empty((0,t_shift,2), dtype=torch.float)\
    
    thresh_rate0 = 1.1
    list_reject = []    
    for i_ID, ID in enumerate(IDs):
        t_start = 0
        if i_ID % 100 == 0:
            print(i_ID)
        # Load data and get label
        if target[i_ID] == 0:
            main_path = path_data[0]
        else:
            main_path = path_data[1]         
        path = main_path+ID
        w = wavio.read(path)
                
        #------------------ cliping the range
        trimm_flg = 0
        thresh_step = 1.05
        thresh_rate = thresh_rate0
        
        while trimm_flg == 0:
            trimm_out = wave_harsh_peaks_all(w.data, t_base, thresh_rate)
            mean_max, list_t, trimmed_t = trimm_out
    
            if len(list_t)==1:
                t_start = 0
                t_stop = len(list_t) - 1
                trimm_flg = 1
            else:
                temp1 = list_t-np.roll(list_t,1)
                ind_list = np.where(min_length < temp1)[0]
                if len(ind_list) > 0:
                    t_start = list_t[ind_list-1]
                    t_stop = list_t[ind_list]-1
                    trimm_flg = 1
                else:
                    t_start = []
                    t_stop = []
#                    thresh_rate = thresh_rate * thresh_step
            
            ECG_splits = 0                              
            for start, stop in zip(t_start, t_stop):
                stop = ((stop - start) // t_shift) * t_shift + start
                t_select = np.array(np.split(trimmed_t[start:stop], range(t_shift,stop-start,t_shift)))
                # t_select = trimmed_t[t_start:t_start+t_length]
        
                windata = w.data[t_select]
                idx_zr = np.where(((windata.max(axis=1) - windata.min(axis=1))==0).sum(axis=1))[0]            
                windata = np.delete(windata, idx_zr, 0)
                ECG_splits += windata.shape[0]
#                if (windata.max(axis=1) - windata.min(axis=1) != 0).all(): # Filter constant values
        #                print (start,stop)
                w_zm = stats.zscore(windata, axis = 1, ddof = 1)
                X = torch.tensor(w_zm).float()
    
                out_IDs = np.append(out_IDs, [(i_ID, start)] * X.shape[0], axis=0).astype(int)
                raw_x = torch.cat((raw_x, X), 0)
                    # raw_x[i_ID,:,:]= X
            
            if ECG_splits >= 15:
                trimm_flg = 1
            else:
                thresh_rate = thresh_rate * thresh_step
                                
        if thresh_rate > thresh_rate0:
            print ("For ID: %d , thresh %2.2f" % (i_ID, thresh_rate))
            list_reject = np.append(list_reject,{'i_ID':i_ID,'thresh':thresh_rate})                
    
    # pickle.dump(list_reject,open("read_data_i_ID.p","wb"))        
    torch.save({'IDs':IDs, 'raw_x':raw_x, 'target':target}, save_file+'_no_split.pt')
    return target, raw_x, IDs, out_IDs, list_reject

def extract_windows(t_win, t_shift, target, raw_x, IDs, out_IDs, save_file, list_reject):

    # length = raw_x.shape[2]
    # n_win = np.floor(length / t_shift) - np.floor(t_win / t_shift)
    # raw_x = raw_x.to(device)
    raw_x_extend = torch.empty([0,raw_x.shape[2],t_win])#.to(device)
    target_extend = np.empty(0)
    data_tag = np.empty(0)

    print("Extracting temporal windows from ECG files..." )
    for i_ID in range(len(IDs)):
        if i_ID % 500 == 0:
            print("data number: {}".format(i_ID))
        idx = np.where(out_IDs[:,0] == i_ID)[0]
        consec_IDs = out_IDs[idx][:,1]
        for c_ID in np.unique(consec_IDs):
            c_idx = idx[consec_IDs == c_ID]
            c_data = raw_x[c_idx]
            c_data = c_data.view(-1,c_data.shape[2])
            c_data = c_data.unfold(0, t_win, t_shift)
            raw_x_extend = torch.cat((raw_x_extend, c_data), 0)
            target_extend = np.append(target_extend, [target[i_ID]] * c_data.shape[0])
            data_tag = np.append(data_tag, [i_ID] * c_data.shape[0])

    torch.save({'raw_x':raw_x_extend,'target':target_extend,'data_tag':data_tag, 'IDs':IDs, 'list_reject':list_reject}, save_file+'.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--window_size",  type=int, default=2048,   help="Size of overlapping windows (default: 2048)")
    parser.add_argument("-s", "--window_shift", type=int, default=400,    help="Shift of overlapping windows (default: 400)")
    parser.add_argument("-p", "--peak_window",  type=int, default=3000,   help="Size of windows to calculate mean maximum amplitude for peak removal (default: 3000)")
    parser.add_argument("out",                                            help="Name of output file")
    args = parser.parse_args()
    main(args)
