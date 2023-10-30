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




fig = plt.figure()
plt.imshow(img, extent = [0, w, 0, h])
plt.show
p = plt.ginput(-1, timeout = 0, mouse_add=3, mouse_pop=2, mouse_stop=9)
p = np.array(p)
plt.close()
n,k = p.shape
A = np.zeros((n,8))


Pw = np.zeros((n,3))

l = float(input('quantidade de quadrados ao longo de x  ---'))
a = float(input('quantidade de quadrados ao longo de y  ---'))
q = float(input('tamanha de cada quadrado  ---'))
cont = 0
Pi = np.zeros((n,2))
Pw = np.zeros((l*a,3))

for i in np.arange(0,n):
    x = float(input('Insira a posião da corrdenada x do ponto ' + str(i +1) + ' ---'))
    y = float(input('Insira a posião da corrdenada y do ponto ' + str(i+ 1) + ' ---'))
    z = float(input('Insira a posião da corrdenada z do ponto ' + str(i+ 1) + ' ---'))

Pi = np.zeros((100,2))
cont = 0
Pi[cont,:] = [p[0,0], p[0,1]]    
dl = np.sqrt((p[0,0] - p[1,0])**2 + ((p[0,1] - p[1,1]))**2)/l
da = np.sqrt((p[0,0] - p[2,0])**2 + ((p[0,1] - p[2,1]))**2)/a
dh = np.sqrt((p[0,0] - p[3,0])**2 + (p[2,1] - p[3,1])**2)/np.sqrt(a^2 + l^2) 

for i in np.arange(1,l):
    for j in np.arange(1,a):
        
        
        Pi[cont + 1,:] = [p[0,0]+i*dl, j*p[0,1]+i*dl]
        
        
        Pi[cont + 2,:] = [i*p[0,0]+da, j*p[0,1]+da]
            
        
        Pi[cont + 3,:] = [i*p[0,0] + dh, j*p[0,1] + dh]           
            
        cont = cont + 3
            
            


P = np.concatenate([Pw,np.ones((n,1))], axis = 1)


for i in np.arange(0,n):

    A[i][:] = np.concatenate([[p[i,0]*P[i,:]],[-p[i,1]*P[i,:]]], axis=1)
    
    

U, S, Vt = np.linalg.svd(A, full_matrices=False)

 
v = Vt[-1,:]

la = np.sqrt(sum((v[0:3])**2))

al = (np.sqrt(sum((v[4:7])**2)))/la

R = np.zeros((3,3))

R[0,:] = v[4:7]/al
R[1,:] = v[0:3]
R[2,:] = np.cross(R[0,:], R[1,:])

# UR, SR, VR = np.linalg.svd(R)
# R = np.dot(UR,np.transpose(VR))

Tx = v[7]/al
Ty = v[3]


RPw = np.dot([R[0,:]],np.transpose(Pw)) + Tx
Aa = np.concatenate([p[:,[0]], np.transpose(RPw)], axis = 1)

b = -1*p[:,0]*np.dot([R[2,:]],np.transpose(Pw))
b = np.transpose(b)

AT = np.transpose(Aa)

Tz, fx = np.dot(np.dot(np.linalg.inv(np.dot(AT,Aa)),AT), b) 

fy = fx/al
T = np.array([[Tx],[Ty],[Tz[0]]])

# ox = w/2-95
# oy = h/2-100


ox = w/2 
oy = h/2 
# fig = plt.figure()
# plt.imshow(img, extent = [0, w, 0, h])
# plt.grid()
# plt.show()
# pc = plt.ginput(-1, timeout = 0, mouse_add=3, mouse_pop=2, mouse_stop=9)
# plt.close()
# pc = np.array(pc)
# X = float(input('Insira a posião da corrdenada x do ponto ---'))
# Y = float(input('Insira a posião da corrdenada y do ponto  ---'))
# Z = float(input('Insira a posião da corrdenada z do ponto  ---'))

if (ox*(sum(R[0,:]*np.array([1*q,0*q,2.5*q]))+T[0])) > 0:
    R[0:2,:] = -1*R[0:2,:]
    T[0:1] = -1*T[0:1]




# f = 12

# sx = 1
# sy = 1


# ox = p[0,0]
# oy = p[0,1]

PWy = np.array([[0],[3],[3]])

PCy = np.dot(R,PWy) + T
xim = -fx*(PCy[0]/PCy[2])+ox
yim = -fy*(PCy[1]/PCy[2])+oy

fig = plt.figure()
plt.scatter(xim,yim, color="black")
plt.scatter(ox,oy, color="black")
plt.imshow(img, extent = [0, w, 0, h])


Px = np.zeros((16,2))
cont = 0
PCx = np.zeros((3,16))
PWx = np.zeros((3,16))
for i in np.arange(1,5):
    for j in np.arange(1,5):
       
        PWx[:,[cont]] = np.array([[i*q],[0],[j*q]])
        PCx[:,[cont]] = np.dot(R,PWx[:,[cont]]) + T
        
        
        xim = -fx*(PCx[0,cont]/PCx[2,cont]) + ox
        yim = -fy*(PCx[1,cont]/PCx[2,cont]) + oy
            
            
        Px[cont,:] = [xim[0],yim[0]]
        
        cont = cont + 1

PWy = np.zeros((3,16))
Py = np.zeros((16,2))
PCy = np.zeros((3,16))
cont = 0        
for i in np.arange(1,5):
    for j in np.arange(1,5):
       
        PWy[:,[cont]] = np.array([[0],[i*q],[j*q]])
        PCy[:,[cont]] = np.dot(R,PWy[:,[cont]]) + T
        
        
        xim = -fx*(PCy[0,cont]/PCy[2,cont]) + ox
        yim = -fy*(PCy[1,cont]/PCy[2,cont]) + oy
            
            
            
        Py[cont,:] = [xim[0],yim[0]]
        
        cont = cont + 1



fig = plt.figure()
plt.scatter(Px[:,0],Px[:,1], color="black")
plt.scatter(Py[:,0],Py[:,1], color="black")
# plt.plot(ox,oy, marker = "v", color="black")
#plt.imshow(img, extent = [-w/2, w/2, -h/2, h/2])
plt.imshow(img, extent = [0, w, 0, h])


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



