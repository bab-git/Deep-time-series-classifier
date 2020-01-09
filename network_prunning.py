#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:05:00 2019

@author: bhossein
Network pruning
"""

import torchvision.models as models

vgg16 = models.vgg16(pretrained = True)