w = 600;
k = 10;

load('c:/Hinkelstien/code_repo/data/raw_x_8K_nofilter_stable.mat');
%%
sinus = cat(1, squeeze(raw_x(1,1,:)), squeeze(raw_x(1,2,:)));
af = cat(1, squeeze(raw_x(8001,1,:)), squeeze(raw_x(8001,2,:)));

% sinus = cat(1, zscore(squeeze(raw_x(1,1,:))), zscore(squeeze(raw_x(1,2,:))));
% af = cat(1, zscore(squeeze(raw_x(8001,1,:))), zscore(squeeze(raw_x(8001,2,:))));

[mp_aa, mpi_aa] = mpx_AB(sinus, sinus, w);
[mp_ab, mpi_ab] = mpx_AB(sinus, af, w);
[mp_bb, mpi_bb] = mpx_AB(af, af, w);
[mp_ba, mpi_ba] = mpx_AB(af, sinus, w);
disp('profiles calculated')
%%
close all

figure
subplot(2,1,1)
plot(sinus)
title('sinus')
subplot(2,1,2)
plot(af)
title('af')

t = 3

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
