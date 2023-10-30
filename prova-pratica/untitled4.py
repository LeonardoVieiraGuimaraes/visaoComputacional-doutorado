# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 10:57:44 2021

@author: leona
"""

import numpy as np
import math as mt

x = 1
y = 1

rr = 100
tt = 200
m = np.zeros((rr, tt))
n = np.zeros((rr, tt))
for r in np.arange(0,rr):
    for t in np.arange(0,tt):
        m[r,t] = x - r*mt.cos(t)
        n[r,t] = y - r*mt.sin(t)
        


import matplotlib.pyplot as plt


for r in np.arange(0,rr):
    plt.plot(n[r,:], m[r,:])
