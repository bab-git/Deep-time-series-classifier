#from __future__ import print_function
import numpy as np

def fib_py(n):
    """Print the Fibonacci series up to n."""
    a, b = 0, 1
    x = np.array([1,2,3])
    while b < n:
        print(b, end=' ')
        a, b = b, a + b

    print(x)