# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:14:40 2021

@author: leona
"""

import cv2
import os


files = os.listdir("./Fotos1")

for cont,file in enumerate(files):
    img = cv2.imread("./Fotos1/" + file)
    imgray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # cv2.resize(imgray, (2202,1301), interpolation = cv2.INTER_AREA)
    cv2.imwrite("./Fotos1/" + "image" + str(cont) + ".jpg",imgray)
    
    
    
    
    
  
    
    