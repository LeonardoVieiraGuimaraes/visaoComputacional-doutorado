# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 10:11:23 2021

@author: leona
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("./Fotos/image3d.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
imgg = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
imgg = np.float32(imgg)


imgs = np.zeros((1280,1280))

row, cols = imgs.shape
imgs[:,:] = 255

for i in np.arange(0,1300,200):
    imgs[:,i] = 0
    imgs[i,:] = 0

plt.figure(2)
plt.imshow(np.uint8(imgs), cmap='gray')

mean = 0.0   # some constant
std = 200    # some constant (standard deviation)
noisy_img = img + np.random.normal(mean, std, img.shape)
# noisy_img_clipped = np.uint8(np.clip(noisy_img, 0, 255) )

# img[dist]=[0,0,255]
plt.figure(1)
plt.imshow(np.uint8(noisy_img))
plt.show()

imgg = imgs
imgg = np.float32(imgg)
Sx = np.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]])
Sy = np.array([[-1, -2, -1], [0, 0, 0],[1, 2, 1]])

Ix = cv2.filter2D(imgg, -1, Sx)
Iy = cv2.filter2D(imgg, -1, Sy)

# Ix = cv2.Sobel(imgg,cv2.CV_64F,1,0,ksize=3)  # x
# Iy = cv2.Sobel(imgg,cv2.CV_64F,0,1,ksize=3)  # y

Ixx = Ix*Ix
Iyy = Iy*Iy
Ixy = Ix*Iy

Sxx = cv2.GaussianBlur(Ixx,(5,5), cv2.BORDER_DEFAULT)
Syy = cv2.GaussianBlur(Iyy,(5,5), cv2.BORDER_DEFAULT)
Sxy = cv2.GaussianBlur(Ixy,(5,5), cv2.BORDER_DEFAULT)

r, c = imgg.shape
k = 0.04

R = (Sxx*Syy - Sxy*Sxy) - k*((Sxx + Syy)**2)

# R = np.zeros((r,c))
# for i in np.arange(0,r):
#     for j in np.arange(0,c):
#         M = [[Sxx[i,j], Sxy[i,j]], [Sxy[i,j],Syy[i,j]]]
#         R[i,j] = np.linalg.det(M) - k*((np.trace(M))**2)

# aa = np.zeros((r,c))
# aa[R>0.04*R.max()]=[255]

# t = 10
# dist = np.full((r, c), True, dtype=bool)
        
# for i in np.arange(0,r):
#     for j in np.arange(0,c):   
        
#         dist[i:i+t, j:j+t] = R[i:i+t, j:j+t] >= (R[i:i+t, j:j+t].max())
        
imgt = imgg
imgt[R>0.04*R.max()]=[255]
# img[dist]=[0,0,255]
plt.figure(0)
plt.imshow(imgt, cmap = 'gray')
plt.show()




mean = 0.0   # some constant
std = 200    # some constant (standard deviation)
noisy_img = img + np.random.normal(mean, std, img.shape)
# noisy_img_clipped = np.uint8(np.clip(noisy_img, 0, 255) )

# img[dist]=[0,0,255]
plt.figure(1)
plt.imshow(np.uint8(noisy_img))
plt.show()

# u = [0,0]
# p0 = [0,0]
# p1 = [x,y]

# p1 = u + t*p0
for a in [-2,2]:
    for b in np.arange(0,10,2):
    
        x = np.linspace(-3, 3, 1000)

        y = a*x + b

        plt.figure(2)
        
        plt.plot(x, y, color='black');


imgs = np.zeros((1280,1280))

row, cols = imgs.shape
imgs[:,:] = 255

for i in np.arange(0,1300,200):
    imgs[:,i] = 0
    imgs[i,:] = 0

plt.figure(2)
plt.imshow(np.uint8(imgs), cmap='gray')