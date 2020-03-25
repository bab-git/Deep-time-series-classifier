# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:44:47 2020

@author: babak
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("fib.pyx")
)