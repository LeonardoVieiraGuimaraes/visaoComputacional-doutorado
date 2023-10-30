# -*- coding: utf-8 -*-
"""
Created on Tue May 14 08:13:04 2019

@author: Leonardo
"""
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import Plotar

def retirad(imagem,r,g,b):
        if r == 0:
            imagem[:,:,0] = 0    #elimina o vermelho
        if g == 0:
            imagem[:,:,1] = 0    #elimina o verde
        if b == 0:
            imagem[:,:,2] = 0    #elimina o azul
        return imagem

especie = 'Italia'
tempo = 0
valor = 30
imgrgb= cv2.imread('Uvas/' + especie + '/' + str(tempo) + '.jpg')
imgLab = cv2.cvtColor(imgrgb,cv2.COLOR_BGR2Lab)

plt.imshow(imgLab[:, :, 0], cmap='Greys')
plt.savefig('Plot/L')
plt.imshow(imgLab[:, :, 1], cmap='PuBuGn_r')
plt.savefig('Plot/a')
plt.imshow(imgLab[:, :, 2], cmap='YlGnBu_r')
plt.savefig('Plot/b')

grayrgb = cv2.cvtColor(imgrgb,cv2.COLOR_RGB2GRAY)
grayLab = cv2.cvtColor(imgLab,cv2.COLOR_RGB2GRAY)

filtrorgb = cv2.GaussianBlur(imgrgb,(101,101),valor)
filtrolab = cv2.GaussianBlur(imgLab,(101,101),valor)

grayfrgb = cv2.cvtColor(filtrorgb,cv2.COLOR_RGB2GRAY)
grayfLab = cv2.cvtColor(filtrolab,cv2.COLOR_RGB2GRAY)

Plotar.plothist(grayrgb, especie + str(tempo) + str(valor)+'grayrgb')
Plotar.plothist(grayLab, especie + str(tempo) + str(valor)+'grayLab')
Plotar.plothist(grayfrgb, especie + str(tempo) + str(valor)+'grayfrgb')
Plotar.plothist(grayfLab, especie + str(tempo) + str(valor)+'grayfLab')

try:
    os.makedirs('Filtro/')
except FileExistsError:
    pass
cv2.imwrite('Filtro/' + especie + '_' + str(tempo) + '_' + str(valor)+'_RGB.jpg', filtrorgb)
cv2.imwrite('Filtro/' + especie + '_' + str(tempo) + '_' + str(valor)+'_Lab.jpg', filtrolab)

try:
    os.makedirs('Plot/')
except FileExistsError:
    pass

Plotar.plot3d(grayrgb[::8,::8], 'RGB')
Plotar.plot3d(grayLab[::8,::8], 'Lab')
Plotar.plot3d(grayfrgb[::8,::8], 'FRGB')
Plotar.plot3d(grayfLab[::8,::8], 'FLab')


L = retirad(imgLab,1,0,0)
a = retirad(imgLab,0,1,0)
b = retirad(imgLab,0,0,1)

cv2.imwrite('Filtro/' + especie + '_' + str(tempo) + '_' + str(valor)+'_L.jpg', np.uint8(L))
cv2.imwrite('Filtro/' + especie + '_' + str(tempo) + '_' + str(valor)+'_a.jpg', a)
cv2.imwrite('Filtro/' + especie + '_' + str(tempo) + '_' + str(valor)+'_b.jpg', b)

plt.close()