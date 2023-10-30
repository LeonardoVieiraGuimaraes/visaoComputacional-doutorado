# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 12:37:52 2021

@author: leona
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("./Fotos/image3d.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
# blockSize=2, aperture size sobel = 3, k =0.04
dst = cv2.cornerHarris(gray,2,3,0.04)
# Threshold for an optimal value, it may vary depending on the image.




gray[dst>0.01*dst.max()]=[255]
plt.figure(1)
plt.imshow(gray, cmap='gray')
plt.show()