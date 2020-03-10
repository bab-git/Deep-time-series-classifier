#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:46:56 2020

@author: bhossein
"""

import sys

for line in sys.stdin:
    if 'Exit' == line.rstrip():
        break
    print('Processing Message from sys.stdin *****{}*****'.format(line))
print("Done")