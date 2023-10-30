# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 00:12:50 2021

@author: leona
"""

import numpy as np

dX = 10 #largura
dY = 20 #altura
dZ = 5

cx = dX/2
cy = dY/2

dx = 5
dy = 20

fx = (dx/dX)*dZ 

fy = (dy/dY)*dZ 

Mint = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])

