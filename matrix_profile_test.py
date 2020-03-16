# import stumpy
import os
from scipy import stats
import wavio
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matrixprofile import matrixProfile, motifs
import numpy as np
#import stumpy

# from numba import cuda
doc_path = '/home/bhossein/BMBF project/code_resources/matrixprofile-ts-master/'

#%%

def read_wav(i_file = None, i_class = None):
    i_file = i_file if i_file else np.random.randint(8000, size = 1).item()
    i_class = i_class if i_class else np.random.randint(2, size = 1)+1    
    if i_class==0:
         main_path = '/vol/hinkelstn/data/FILTERED/sinus_rhythm_8k/'
#        main_path = '/vol/hinkelstn/data/ecg_8k_wav/sinus_rhythm_8k/'
    else:  
         main_path = '/vol/hinkelstn/data/FILTERED/atrial_fibrillation_8k/'
#        main_path = '/vol/hinkelstn/data/ecg_8k_wav/atrial_fibrillation_8k/'

    list_f = os.listdir(main_path)
    file = list_f[i_file]
    path = main_path+file

    w = wavio.read(path)
    return w

#%%   
plt.close('all')
#def main():
#clrcds = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#2a7700','#bcbd22','#17becf']
clrcds = ['g', 'r', 'c', 'y', 'k']
mrkcds = ['o', 'v' ,'s', '1', '*', 'p']
 
radius, i_file, i_class, ex_zone = [None] *4

i_file = 1432
i_class = 0
k = 6
window_size = 30 # Approximately, how many data points might be found in a pattern
#ex_zone = window_size
#radius = 1.5

ts = []

wav = read_wav(i_file = i_file, i_class= i_class).data.astype(float)
wav = wav[500:]
fig, ax = plt.subplots(3,1,sharex=True,figsize=(10,5))
for i in range(2):
    ax[i].plot(np.arange(len(wav)),wav[:,i], label="Synthetic Data")
ax[2].plot(np.arange(len(wav)),wav[:,1] - wav[:,0], label="Synthetic Data")

data = pd.read_csv(doc_path+'docs/examples/rawdata.csv')

#ts = wav[:,1] - wav[:,0]
#ts = wav[:,0]
#ts = np.abs(np.sin(np.linspace(0, np.pi*5, 5*25+1)))
#ts = np.append(ts, np.abs(np.sin(np.linspace(0, np.pi, int(1/1.5*25+3))))[1:])
#ts = np.append(ts, -np.abs(np.sin(np.linspace(0, np.pi*3, 3*24+1)))[1:])
ts = data.data.values
#fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True,figsize=(17,9))
#ax1.plot(np.arange(len(ts)),ts, label="Synthetic Data")
#ax1.set_ylabel('Signal', size=22)

ts = stats.zscore(ts)

# gpu_device = 1

#matrix_profile = stumpy.gpu_stump(ts, m=window_size, device_id=gpu_device)
#mp = matrixProfile.scrimp_plus_plus(ts, window_size)
#mp = matrixProfile.stomp(ts,window_size)
mp_adj = np.append(mp[0],np.zeros(window_size-1)+np.nan)


#%%
ex_zone = window_size
radius = 2

mtfs, dsts = motifs.motifs(ts, mp, max_motifs= k, radius = radius, ex_zone = ex_zone )
#mtfs, dsts = motifs.motifs(ts, mp, max_motifs= k, radius = radius)

fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True,figsize=(17,9))
ax1.plot(np.arange(len(ts)),ts, label="Synthetic Data")
ax1.set_ylabel('Signal', size=22)

#Plot the Matrix Profile
ax2.plot(np.arange(len(mp_adj)),mp_adj, label="Matrix Profile", color='red')
ax2.set_ylabel('Matrix Profile', size=22)
ax2.set_xlabel('Sample', size=22)
ax2.set_title('window_size = {}'.format(window_size))

colors = np.zeros_like(ts)
for i, mtf in enumerate(mtfs):
    for j in mtf:
        colors[j:j+window_size] = i+1
print(np.asarray(mtfs).shape)

ax3.plot(np.arange(len(ts)),ts, label="Synthetic Data", linestyle = '--', linewidth  = 1)
ax3.set_ylabel('Signal+MP', size=22)

for ind in range(len(mtfs)):
#        plt.subplot(2,1,2)
    mtf = mtfs[ind]
#    clr = np.unique(colors)[ind].astype(int)
    i_lgnd = 0
    for ind_j in range(len(mtf)):
#        t = 
        t = range(mtf[ind_j],mtf[ind_j]+window_size)
        if i_lgnd ==0:
            ax3.plot(t, ts[t], color = clrcds[ind], linewidth  = 3, label = 'MP: '+str(ind))
            i_lgnd =1
        else:
            ax3.plot(t, ts[t], color = clrcds[ind], linewidth  = 3)
        ax3.scatter(mtf[ind_j], ts[mtf[ind_j]], color = clrcds[ind], marker = mrkcds[ind], linewidth = 3)
ax3.legend()        


assert 1==2
#%%
def _applyExclusionZone(prof, idx, zone):
    start = int(max(0, idx - zone))
    end = int(idx + zone + 1)
    prof[start:end] = np.inf

motifs = []
distances = []
try:
    mp_current, mp_idx = mp
except:
    raise ValueError("argument mp must be a tuple")
mp_current = np.copy(mp_current)

#if len(ts) <= 1 or len(mp_current) <= 1 or max_motifs == 0:
#    return [], []

m = len(ts) - len(mp_current) + 1
#    print(m)
if m <= 1:
    raise ValueError('Matrix profile is longer than time series.')
if ex_zone is None:
    ex_zone = m / 2
#    print(ex_zone)
for j in range(max_motifs):
    # find minimum distance and index location
    min_idx = mp_current.argmin()
    motif_distance = mp_current[min_idx]
    if motif_distance == np.inf:
        print('return motifs, distances')
        assert 1==2
    if motif_distance == 0.0:
        motif_distance += np.finfo(mp_current.dtype).eps

    motif_set = set()
    initial_motif = [min_idx]
    pair_idx = int(mp[1][min_idx])
    if mp_current[pair_idx] != np.inf:
        initial_motif += [pair_idx]

    motif_set = set(initial_motif)

    prof, _ = distanceProfile.massDistanceProfile(ts, initial_motif[0], m)

    # kill off any indices around the initial motif pair since they are
    # trivial solutions
    for idx in initial_motif:
        _applyExclusionZone(prof, idx, ex_zone)
    # exclude previous motifs
    for ms in motifs:
        for idx in ms:
            _applyExclusionZone(prof, idx, ex_zone)

    # keep looking for the closest index to the current motif. Each
    # index found will have an exclusion zone applied as to remove
    # trivial solutions. This eventually exits when there's nothing
    # found within the radius distance.
    prof_idx_sort = prof.argsort()

    for nn_idx in prof_idx_sort:
        if n_neighbors is not None and len(motif_set) >= n_neighbors:
            break
        if prof[nn_idx] == np.inf:
            continue
        if prof[nn_idx] < motif_distance * radius:
            motif_set.add(nn_idx)
            _applyExclusionZone(prof, nn_idx, ex_zone)
        else:
            break

    for motif in motif_set:
        _applyExclusionZone(mp_current, motif, ex_zone)

    if len(motif_set) < 2:
        continue
    motifs += [list(sorted(motif_set))]
    distances += [motif_distance]



