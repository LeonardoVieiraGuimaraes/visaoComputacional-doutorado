# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 00:12:50 2021

@author: leona
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#########################################################


#img = plt.imread(dir + "Book.jpg")
img = cv2.imread("./Foto0/Book.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
h,w,d = img.shape
#Tamanho da imagem em cm
dX = 17 #largura
dY = 24 #altura
dZ = 20 #distãncia da imagem com a camera

#Tamanho da imagem em pixel
dx = 360
dy = 512


# Centro da Imagem
cx = w/2
cy = h/2-49

#Distancia Focal
fx = (dx/dX)*dZ 
fy = (dy/dY)*dZ 

# Parametros Intricicos
Mint = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])

##########################################################

xi = np.zeros((1,9))
              
yi = np.zeros((1,9))

Pw = np.zeros((3,9))
cont = 0

for i in np.arange(-1,2): 
    for j in np.arange(-1,2):
        
        Pw[:,[cont]] = np.array([[8.5*i],[12*j],[20]])
        pin = Mint@Pw[:,[cont]]
        xi[0,cont] = pin[0,0]/pin[2,0]
        yi[0,cont] = pin[1,0]/pin[2,0] 
        cont = cont +1

plt.plot(xi,yi, marker = "v", color="black")

plt.imshow(img, extent = [0, w, 0, h])
plt.show()




  






