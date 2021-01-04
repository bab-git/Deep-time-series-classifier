#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:15:31 2020

@author: bhossein
"""

from torch._ops import ops


qfn = torch.nn.quantized.QFunctional()
a = torch.rand(10)
b = torch.rand(10)
#%%
min_a = 0
min_b = 0
max_a = 3*max(a)
max_b = 3*max(b)
qmax = 255
qmin = 0
scale_a = (max_a - min_a) / (qmax - qmin)
zpt_a = int(qmin - min_a / scale_a)
zpt_a = 10
scale_b = (max_b - min_b) / (qmax - qmin)
zpt_b = int(qmin - min_b / scale_b)
zpt_b = 12
a_quant = torch.quantize_per_tensor(a, scale_a, zpt_a, torch.qint8)
b_quant = torch.quantize_per_tensor(b, scale_b, zpt_b, torch.qint8)

#print(qfn.add(a_quant, b_quant))
c= ops.quantized.add(a_quant, b_quant, scale = scale_a, zero_point = 5)
print(c)
print(c.int_repr())
a_f = scale_a*(a_quant.int_repr().data-zpt_a)
b_f = scale_a*(b_quant.int_repr().data-zpt_b)
(a_f+b_f)/scale_a+5