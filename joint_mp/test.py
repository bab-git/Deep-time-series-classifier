# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:01:03 2020

@author: babak
"""
import numpy as np
#import fib
import time

def sum_test(n,m):
    """Print the Fibonacci series up to n."""
#    a, b = 0, 1
    x = np.zeros(len(n)-100)
    for i in range(len(x)):        
        x[i] = np.sum(n[i:i+20])*m[i+5]
#    while b < n:
#        print(b, end=' ')
#        a, b = b, a + b
    return x    

def sum_iter(na,nb):
    x = np.sum(na)*nb
    return x


#    print(x)

N = 1000
n = np.random.random((N,1))
m = np.random.random((N,1))
t0 = time.time()        
x = sum_test(n,m)
t1 = time.time()-t0        
print (t1)

a = np.random.random((N,1))
b = np.random.random((N,1))
t0 = time.time()        
#y = fib.sum_test(a,b)
#na = [n[i:i+20] for i in range(len(n)-100)]
na = map(lambda i:n[i:i+20], range(len(n)-100))
ma = np.roll(m,-5)
y = (map(sum_iter, na, ma))
t2 = time.time()-t0        
print (t2)

#y==x