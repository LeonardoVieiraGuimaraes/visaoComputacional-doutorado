# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 23:48:44 2021

@author: leona
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

src = cv.imread('./circulo.png')

dst = cv.Canny(src, 50, 200, None, 3)

cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

m, n = np.shape(dst)




R = 100
T = 100
A = np.zeros((R,T))

d = int(np.sqrt(m**2 + n**2))
dr = 2*d/R
dt = pi/R


for i in np.arange(0,m):
    for j in np.arange(0,n):
        
        for t in np.arange(0,t):
            
            r = i*cos(t*dt) + j*sin(t*dt)
            r = round((r+d)/dr) +1
            if (r>0 & r<R):
                A[r,t] = A(r,t) + 1
            


plt.figure()
plt.imshow(cdst)