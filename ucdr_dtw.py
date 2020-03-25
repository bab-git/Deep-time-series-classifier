# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 19:23:27 2020

@author: babak
"""

import _ucrdtw
import numpy as np
import matplotlib.pyplot as plt

data = np.cumsum(np.random.uniform(-0.5, 0.5, 10000))
query = np.cumsum(np.random.uniform(-0.5, 0.5, 100))
loc, dist = _ucrdtw.ucrdtw(data, query, 0.05, True)
query = np.concatenate((np.linspace(0.0, 0.0, loc), query)) + (data[loc] - query[0])

plt.figure()
plt.plot(data)
plt.plot(query)
plt.show()