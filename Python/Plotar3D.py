# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:55:03 2019

@author: Leonardo
"""

import Plotar
import cv2

img = cv2.imread('C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/Vitoria1/V_0h_1.jpg')


def retirad(imagem,r,g,b):
        if r == 0:
            imagem[:,:,2] = 0    #elimina o vermelho
        if g == 0:
            imagem[:,:,1] = 0    #elimina o verde
        if b == 0:
            imagem[:,:,0] = 0    #elimina o azul   
        return imagem



filtro = cv2.GaussianBlur(img,(101,101),0)
        
#imgYCrCb = cv2.cvtColor(filtro,cv2.COLOR_BGR2YCrCb)
        
#imgHLS = cv2.cvtColor(filtro,cv2.COLOR_BGR2HLS)
        
              
#img1 = retirad(imgYCrCb,0,1,0)
gray = cv2.cvtColor(filtro,cv2.COLOR_BGR2GRAY)

Plotar.plot3d(gray)