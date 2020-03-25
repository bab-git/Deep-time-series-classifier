# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 03:11:40 2020

@author: babak
"""
import matplotlib.pyplot as plt
from default_modules import *

import wavio
#import os

os.chdir('C:\Hinkelstien\code_repo\joint_mp')

from matrixprofile import matrixProfile, fluss
from scipy import signal


from mpx_AB import mpx_AB



w = 600;
k = 10;

#data_dir = '/vol/hinkelstn/codes/'

#load_ECG = torch.load('/vol/hinkelstn/codes/raw_x_8K_nofilter_stable.pt')
load_ECG = torch.load('c:/Hinkelstien/code_repo/data/raw_x_8K_nofilter_stable.pt')
raw_x = load_ECG['raw_x']

#load_ECG = pickle.load(open(,'rb'))
#%%
sinus = np.concatenate((raw_x[0,0,:], raw_x[0,1,:]))
af = np.concatenate((raw_x[8000,0,:], raw_x[8000,1,:]))

sinus_dn =  signal.resample(sinus, int(len(sinus)/4))
af_dn =  signal.resample(af, int(len(af)/4))

w = 600
[mp_ab, mpi_ab]  = matrixProfile.stomp(tsA= sinus,m= w, tsB= af)
[mp_aa, mpi_aa] = matrixProfile.stomp(tsA= sinus,m= w, tsB= sinus)
mp_aa2 = matrixProfile.stomp(tsA= sinus,m= w)
mp_aa2 = mp_aa2[0]
#% sinus = cat(1, zscore(squeeze(raw_x(1,1,:))), zscore(squeeze(raw_x(1,2,:))));
#% af = cat(1, zscore(squeeze(raw_x(8001,1,:))), zscore(squeeze(raw_x(8001,2,:))));

[mp_aa, mpi_aa] = mpx_AB(sinus, sinus, w)
[mp_ab, mpi_ab] = mpx_AB(sinus, af, w)
[mp_bb, mpi_bb] = mpx_AB(af, af, w)
[mp_ba, mpi_ba] = mpx_AB(af, sinus, w)
print('profiles calculated')
#%%
plt.close('all')

plt.figure()
plt.subplot(2,1,1)
plt.plot(sinus)
plt.title('sinus')
plt.subplot(2,1,2)
plt.plot(af)
plt.title('af')

#t = 3

p_a = mp_ab - mp_aa;
p_b = mp_ba - mp_bb;

p_a(8000-w:8000+w) = 0;
p_b(8000-w:8000+w) = 0;

figure
subplot(4,1,1)
plot(mp_ab)
title('mp_ab')
subplot(4,1,2)
plot(mp_aa)
title('mp_aa')
subplot(4,1,3)
plot(p_a)
title('p_a')
subplot(4,1,4)
hold off
plot(sinus)
shapelets_a = shapelets_extract(sinus,p_a,w,k);

figure
subplot(3,1,1)
plot(mp_ba)
title('mp_ba')
subplot(3,1,2)
plot(mp_bb)
title('mp_bb')
subplot(3,1,3)
plot(p_b)
title('p_b')
subplot(4,1,4)
hold off
plot(af)
shapelets_b = shapelets_extract(af,p_b,w,k);

%% Search
shplt = shapelets_a(1,:)';
dist3 = abs(MASS_V3_me(sinus, shplt ,2*w));
% dist2 = abs(MASS_V2(sinus, shplt));

figure
subplot(3,1,1)
plot(sinus)
subplot(3,1,2)
plot(shplt)
subplot(3,1,3)
plot(dist2)
    

%%
