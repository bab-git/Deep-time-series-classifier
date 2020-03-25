# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:44:06 2020

@author: babak
"""

from __future__ import print_function

def fib(n):
    """Print the Fibonacci series up to n."""
    a, b = 0, 1
    while b < n:
        print(b, end=' ')
        a, b = b, a + b

    print()