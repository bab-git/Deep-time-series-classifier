# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 19:33:33 2020

@author: babak
"""
import numpy as np

from itertools import accumulate


#% Functions here are based on the work in 
#% Ogita et al, Accurate Sum and Dot Product
#%%
def muinvn(a,w):
    #% results here are a moving average and stable inverse centered norm based
    #% on Accurate Sum and Dot Product, Ogita et al
    
#    mu = [i/w for i in sum2s(a,w)]
    mu = sum2s(a,w)/w
    
#    t0 = time.time()
           
    sig_in = map(lambda i:a[i:i+w]-mu[i], range(len(mu)))
    
    sig_out = map(sq2s, sig_in)
    sig = map(lambda i: 1/np.sqrt(i), sig_out)

#    t1 = time.time()-t0
#    print(t1)

#    sig_out = list(sig_out)
#    t1b = time.time()-t0
#    print(t1b)
    
#    t0 = time.time()
#    
#    sig = np.zeros(len(a) - w + 1)
#    for i in range(len(mu)):
#        sig[i] = sq2s(a[i : i + w] - mu[i])
#            
#    sig = [1/np.sqrt(i) for i in sig]
#    t2 = time.time()-t0
#    print(t2)
#    sig == list(sig_out)
    return mu,sig
#%%    
def sq2s(a):
#    h = [i * i for i in a]
#    c = [((2**27) + 1) * i for i in a]  # <-- can be replaced with fma where available
#    a1 = [(ic - (ic - ia)) for ic,ia in zip(c,a)]
#    a2 = [ia - ia1 for ia,ia1 in zip(a,a1)] 
#    a3 = [ia1*ia2 for ia1,ia2 in zip(a1,a2)]
#    r = [ia2*ia2 - (((ih - ia1*ia1) - ia3) - ia3) for ia1,ia2,ih,ia3 in zip(a1,a2,h,a3)]

    h = np.square(a) 
    c = 134217729*a   # <-- can be replaced with fma where available
    a1 = (c - (c - a))
    a2 = a-a1
    a3 = a1*a2
    r = a2*a2 - (((h - a1*a1) - a3) - a3) 

    p = h[0] 
    s = r[0] 
    for i in range(1, len(a)):
        x = p + h[i] 
        z = x - p 
        s = s + (((p - (x - z)) + (h[i] - z)) + r[i]) 
        p = x 

    res = p + s
    return res
    
#%%    
def sum2s(a,w):
    res = np.zeros(len(a) - w + 1)
    p = a[0] 
    s = 0 
    for i in range(1,w):
        x = p + a[i] 
        z = x - p 
        s = s + ((p - (x - z)) + (a[i] - z)) 
        p = x
    
    res[0] = p + s
    for i in range(w, len(a)):
        x = p - a[i - w]
        z = x - p
        s = s + ((p - (x - z)) - (a[i - w] + z))
        p = x
        
        x = p + a[i]
        z = x - p
        s = s + ((p - (x - z)) + (a[i] - z))
        p = x
        
        res[i - w + 1] = p + s
    
    return res
#%%  
dfa = map(lambda ib:df[ia+ib], range(mx))
dga = map(lambda ib:dg[ia+ib], range(mx))
invnaa = map(lambda ib:invna[ia+ib], range(mx))
mpa = map(lambda ib:mp[ia+ib], range(mx))

c = c + df[ib + ia] * dy[ib] + dg[ib + ia] * dx[ib];

startVal =  18
myList   =  [0,0,0,1,0,2]
accumulator = reduce((lambda x, y: ( x + [y + (0 if len(x) == 0 else x[-1])])), myList, [])  
[startVal + v for v in accumulator]


c_cmp = map(mp_sub, dfa, dy, dga, dx,invnaa, invnb)
def mp_sub(dfa,dy,dga,dx,invnaa,invnb, c = c):
    c = c + dfa * dy + dga * dx
    c_cmp = c * invnaa * invnb;
#    if c_cmp > mpa[ib + ia]:
#        mp[ib + ia] = c_cmp;
#        mpi[ib + ia] = ib;
    return c_cmp

def mpx_AB(a, b, w):
#    % matrix profile code 
#    % inputs are 2 time series and a window length
#    % It returns a profile with the distance and correspoding index between each window 
#    % in a and its nearest neighbor in b.
    if not isinstance(a,np.ndarray):
        a = np.array(a)
    
    if not isinstance(b,np.ndarray):
        b = np.sarray(b)
    
    [mua, invna] = muinvn(a,w)
    [mub, invnb] = muinvn(b,w)
    
#    % difference equations have 0 as their first entry. This simplifies index
#    % calculations slightly and allows us to avoid special "first line"
#    % handling.
    
#    % This is a basic reference implementation, written to avoid the need to
#    % compile anything when distributed. 
    enda = len(a)
    endb = len(b)
    endma = len(mua)
    endmb = len(mub)
#    f_sub = lambda a
#    a_w = a[w : enda-1]
    t_df = [(1/2)*(i-j) for i,j in zip(a[w : enda], a[: enda - w])]
    df = np.append(0, t_df);
    
#    dg = [0; (a(1 + w : end) - mua(2 : end)) + (a(1 : end - w) - mua(1 : end - 1))];    
    dg = [ i-j+c-d for (i,j,c,d) in zip(a[w:enda], mua[1:endma], a[: enda-w], mua[:endma-1])]
    dg = np.append(0,dg)

#    dx = [0; (1/2)*(b(1 + w : end) - b(1 : end - w))];
    dx = [(1/2)*(b1 - b2) for b1,b2 in zip(b[w : endb] , b[: endb - w])];
    dx = np.append(0,dx)
    
#    dy = [0; (b(1 + w : end) - mub(2 : end)) + (b(1 : end - w) - mub(1 : end - 1))];
    dy = [ b1-m1+b2-m2 for (b1,m1,b2,m2) in zip(b[w : endb] ,mub[1 : endmb]\
          ,b[:endb - w], mub[: endmb - 1])]
    dy = np.append(0,dy)
    
    amx = len(a) - w + 1;
    bmx = len(b) - w + 1;
    mp = [-1]*amx
    mpi = [np.nan]*amx
    a = np.array(a)
    b = np.array(b)
    for ia in range(amx):
        print(ia)
        mx = min(amx - ia, bmx);
        c = sum((a[ia : ia + w] - mua[ia]) * (b[:w] - mub[0]))
#        c = sum(np.array(a[ia : ia + w] - mua[ia]) * np.array(b[:w] - mub[0]));
                       
        c_i = accumulate(map(lambda ib: c if ib ==-1 else df[ib + ia] * dy[ib] + dg[ib + ia] * dx[ib],range(-1,mx)), lambda x,y:x+y)    
        c_cmp = list(map(lambda ci,ib:ci*invnab*invnb[ib], c_i, range(mx)))
        for ib in range(mx):
#            c = c + df[ib + ia] * dy[ib] + dg[ib + ia] * dx[ib];
            c_cmp = c * invna[ib + ia] * invnb[ib];
            if c_cmp > mp[ib + ia]:
                mp[ib + ia] = c_cmp;
                mpi[ib + ia] = ib;
      
       
        
        
        
        for ib in range(mx):
            c = c + df[ib + ia] * dy[ib] + dg[ib + ia] * dx[ib];
            c_cmp = c * invna[ib + ia] * invnb[ib];
            if c_cmp > mp[ib + ia]:
                mp[ib + ia] = c_cmp;
                mpi[ib + ia] = ib;
            
    
    for ib in range(bmx):
        mx = min(bmx - ib, amx);
        c = sum(np.array(b[ib : ib + w ] - mub[ib]) * np.array(a[:w] - mua[0]))
        for ia in range(mx):
            c = c + df[ia] * dy[ib + ia] + dg[ia] * dx[ib + ia]
            c_cmp = c * invna[ia] * invnb[ib + ia];
            if c_cmp > mp[ia]:
                mp[ia] = c_cmp;
                mpi[ia] = ia + ib - 1;
            
    if(any([i>1 for i in mp])):
        print('possible precision loss due to rounding') 
    
    mp = [np.sqrt(2 * w * (1 - min(1, imp))) for imp in mp]
    
    return mp, mpi

#%%
#def sum2s_v2(a,w):
#    res = zeros(length(a) - w + 1, 1) 
#    accum = a(1) 
#    resid = 0 
#    for i = 2 : w
#        m = a[i] 
#        p = accum 
#        accum = accum + m 
#        q = accum - p 
#        resid = resid + ((p - (accum - q)) + (m - q)) 
#    end
#    res(1) = accum + resid;
#    for i = w + 1 : length(a)
#        m = a(i - w);
#        n = a[i];
#        p = accum - m;
#        q = p - accum;
#        r = resid + ((accum - (p - q)) - (m + q));
#        accum = p + n;
#        t = accum - p;
#        resid = r + ((p - (accum - t)) + (n - t));
#        res(i - w + 1) = accum + resid;
#    end
#    return res

#%%    
#def TwoSquare(a):
#    x = a .* a 
#    c = ((2^27) + 1) .* a 
#    a1 = (c - (c - a)) 
#    a2 = a - a1 
#    a3 = a1 .* a2 
#    y = a2 .* a2 - (((x - a1 .* a1) - a3) - a3) 
#    
#    return x,y
