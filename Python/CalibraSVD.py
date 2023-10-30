# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:57:14 2021

@author: leona
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


dir = "C:/Users/leona/Desktop/Doutorado/VisaoComputacional/Quiz2/Fotos2/"

img = cv2.imread(dir + '20210618_100500.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

h,w,d = img.shape
q = 30
n = 4
A = np.zeros((n,8))

Pi = np.zeros((n,2))
Pw = np.zeros((n,3))

fig = plt.figure()
plt.imshow(img, extent = [0, w, 0, h])
p = plt.ginput(n, timeout = 0, mouse_add=3, mouse_pop=1, mouse_stop=2)
plt.show
plt.close()
for i in np.arange(0,n):
   
    
    xi = p[i][0]
    yi = p[i][1]
    
    # Pi[i,:] = np.array([xi,yi])
    #Ponto do eixto plano xz 
    
    x = int(input('Insira a posião da corrdenada x do ponto ' + str(i +1) + ' ---'))
    y = int(input('Insira a posião da corrdenada y do ponto ' + str(i+ 1) + ' ---'))
    z = int(input('Insira a posião da corrdenada z do ponto ' + str(i+ 1) + ' ---'))
    
    Pw[i,:] = np.array([x*q,y*q,z*q])   
    
    P = np.concatenate([Pw[i,:],[1]])
        
    A[i][:] = np.concatenate([[xi*P],[-yi*P]], axis=1)
    
    

_, _, Vt = np.linalg.svd(A, full_matrices=False)

v = Vt[-1,:]

la = np.sqrt(sum((v[0:3])**2))

al = (np.sqrt(sum((v[4:8])**2)))/la

R = np.zeros((3,3))

R[0,:]  = v[4:7]/al
R[1,:] = v[0:3]

R[2,:] = np.cross(R[0,:], R[1,:])



RPw = np.dot([R[0,:]],np.transpose(Pw))
Aa = np.concatenate([np.transpose([P[:,0]]), np.transpose(RPw)], axis = 1)

b = Pi[:,0]*np.dot([R[2,:]],np.transpose(Pw))


AT = np.transpose(Aa)

Ata = np.dot(np.linalg.inv(np.dot(AT,Aa)),AT)


Tz, fx = np.dot(Ata, np.transpose(b)) 
fy = al*fx
T = np.array([v[7:8]/al,v[3:4],Tz])

fig = plt.figure()
plt.imshow(img)
plt.grid()
plt.show()
p = plt.ginput(1, timeout = 0)
plt.close()

x = int(input('Insira a posião da corrdenada x do ponto ' + str(i +1) + ' ---'))
y = int(input('Insira a posião da corrdenada y do ponto ' + str(i+ 1) + ' ---'))
z = int(input('Insira a posião da corrdenada z do ponto ' + str(i+ 1) + ' ---'))

if x*(sum(R[0,:]*np.array([x*q,y*q,z*q]))+T[0]) > 0:
    R[0:1,:] = -1*R[0:2,:]
    T[0:1] = -1*T[0:1]

PW = np.array([[90],[0],[90]])

Xc, Yc, Zc = np.dot(R,PW) + T

h,w,d = img.shape

f = 26

sx = fx*f
sy = fy*f

ox = w/2
oy = h/2

xim = (-f/sx[0])*(Xc[0]/Zc[0]) + ox
yim = (-f/sy[0])*(Yc[0]/Zc[0]) + oy


fig = plt.figure()
plt.plot(xim,yim, marker = "v", color="black")
plt.plot(ox,oy, marker = "v", color="black")
# plt.imshow(img, extent = [-w/2, w/2, -h/2, h/2])
plt.imshow(img, extent = [0, w, 0, h])
plt.grid()
plt.show()
# if x*(r11*Xw + r12*Yw + r13*Zw + Tx) > 0: 
    

# A = [[xi], R[0,:]**P]



# A = np.array([[13, -5], [-5, 13]])





# ##Coordenadas do ponto no mundo

# # Pi = [Xw Yw Zw]

# ##Coordenadas do ponto na camera

# # R = [Xc Yx Zc]

# ##Coordenadas da imagem projetada

# # Pp = [x y]

# ##Transladar

# #T = [Tx Ty Tz]

# # Pc = RTPw = Mext Pw
# # p = MintPc

# # p = Mint Mext Pw



