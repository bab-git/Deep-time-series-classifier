# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:44:06 2020

@author: babak
"""

#from __future__ import print_function

import numpy as np

def sum_test(n,m):
    """Print the Fibonacci series up to n."""
#    a, b = 0, 1
    x = np.zeros(len(n)-100)
    for i in range(len(x)):        
        x[i] = np.sum(n[i:i+20])*m[i+5]
#    while b < n:
#        print(b, end=' ')
#        a, b = b, a + b

#    print(x)