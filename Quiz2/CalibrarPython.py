# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:57:14 2021

@author: leona
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Realização das medicações dos pontos na imagem com o referêncial do mundo 
img = cv2.imread('./Fotos2/image3d.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

h,w,d = img.shape
q = 30

fig = plt.figure()
plt.imshow(img, extent = [0, w, 0, h])
plt.show
p = plt.ginput(-1, timeout = 0, mouse_add=3, mouse_pop=2, mouse_stop=9)
p = np.array(p)
plt.close()
n,k = p.shape

Pw = np.zeros((n,3))
cont = 0

l = 5
a = 5
h = np.sqrt(l**2 + a**2)

# primeiro com o segundo ponto


u = np.array([p[0,0] - p[1,0], p[0,1] - p[1,1]])

# primeiro com o terceiro ponto
v = np.array([p[0,0] - p[2,0], p[0,1] - p[2,1]])

uv = u + v

# # primeiro com o quarto ponto
# dx3 = (p[0,0] - p[3,0])/h
# dy3 = (p[0,1] - p[3,1])/h

cont = 0
Pp = np.zeros((l,a,2))

Pp[0,0,:] = p[0,:]



# (j*(dx2/dx1)/a)
# (j*(dy2/dy1)/a)



for i in np.arange(0,l):
    for j in np.arange(0,a):
    

        
        Pp[i+1, j+1,:] = [Pp[j,i,:] + v]
        Pp[i+1, j+1,:] = [Pp[j,i,:] + u]
    
    
       
        
        # Pp[cont+1,:] = [p[3,0] + i*dx2/l, p[3,1] + i*dy2/l]
    
        # Pp[cont+1,:] = [p[0,0] + a*dx2, p[0,1] + a*dy2]
        
        # Pp[cont+2,:] = [p[0,0] + dx3, p[0,1] + dy3]
        
   

plt.scatter(Pp[:,0],Pp[:,1], color="black")

for i in np.arange(0,n):

    x = float(input('Insira a posião da corrdenada x do ponto ' + str(i +1) + ' ---'))
    y = float(input('Insira a posião da corrdenada y do ponto ' + str(i+ 1) + ' ---'))
    z = float(input('Insira a posião da corrdenada z do ponto ' + str(i+ 1) + ' ---'))
    
    Pw[i,:] = np.array([x*q,y*q,z*q])  
                          
P = np.concatenate([Pw,np.ones((n,1))], axis = 1)

######################################################################################
#Tendo estabelecido as correspondências N entre a imagem e os pontos do mundo, 
#calcule o SVD de A. A solução é a coluna de V correspondente ao menor valor 
#singular de A

A = np.zeros((n,8))
for i in np.arange(0,n):
    A[i][:] = np.concatenate([[p[i,0]*P[i,:]],[-p[i,1]*P[i,:]]], axis=1)
      

_, _, Vt = np.linalg.svd(A, full_matrices=False)
v = Vt[-1,:]

####################################################################################
#Deteminando o valor de lambida e alfa 


la = np.sqrt(sum((v[0:3])**2))
al = (np.sqrt(sum((v[4:7])**2)))/la

####################################################################################
#Recupere as duas primeiras linhas de R e os dois primeiros componentes de T, 
#Calcula a terceira linha de R como o produto vetorial das duas primeiras linhas 

R = np.zeros((3,3))

R[0,:]  = v[4:7]/al
R[1,:] = v[0:3]
R[2,:] = np.cross(R[0,:], R[1,:])

Tx = v[7]/al
Ty = v[3]

###############################################################################
#Congirugando o Valor de A e b para estimar os valores fx e Tz, obtendo T
#e também obtendo o fy. 

RPw = np.dot([R[0,:]],np.transpose(Pw)) + Tx
Aa = np.concatenate([p[:,[0]], np.transpose(RPw)], axis = 1)

b = -1*p[:,0]*np.dot([R[2,:]],np.transpose(Pw))
b = np.transpose(b)

AT = np.transpose(Aa)
Tz, fx = np.dot(np.dot(np.linalg.inv(np.dot(AT,Aa)),AT), b) 

fy = fx/al
T = np.array([[Tx],[Ty],[Tz[0]]])
###############################################################################
ox = 0
oy = 0

if (ox*(sum(R[0,:]*np.array([1*q,0*q,2.5*q]))+T[0])) > 0:
    R[0:2,:] = -1*R[0:2,:]
    T[0:1] = -1*T[0:1]
###############################################################################
#Com as Matrizes R, T, fx, fy, a calibração está realizando podendo estimar os pontos
#do mundo com os pontos da imagem

point = 10
Po = np.zeros((2*point**2,2))
PC = np.zeros((3,2*point**2))
PW = np.zeros((3,2*point**2))
cont = 0

for i in np.arange(1,point):
    for k in np.arange(1, point):
    
        PW[:,[cont]] = np.array([[i*q],[0],[k*q]])
        PC[:,[cont]] = np.dot(R,PW[:,[cont]]) + T
        xim = -fx*(PC[0,cont]/PC[2,cont]) + ox
        yim = -fy*(PC[1,cont]/PC[2,cont]) + oy
        Po[cont,:] = [xim[0],yim[0]]
        cont = cont + 1
        
        PW[:,[cont]] = np.array([[0],[i*q],[k*q]])
        PC[:,[cont]] = np.dot(R,PW[:,[cont]]) + T
        xim = -fx*(PC[0,cont]/PC[2,cont]) + ox
        yim = -fy*(PC[1,cont]/PC[2,cont]) + oy
        Po[cont,:] = [xim[0],yim[0]]
        cont = cont + 1
        
fig = plt.figure()
plt.scatter(Po[:,0],Po[:,1], color="black")
plt.imshow(img, extent = [0, w, 0, h])

