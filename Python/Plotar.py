# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 13:01:29 2018

@author: Leonardo
"""


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

def sep(img):
    m, n, d = np.shape(img)
    X = np.zeros([m, n])
    Y = np.zeros([m, n])
    Z = np.zeros([m, n])
    X = img[:, :, 0]
    Y = img[:, :, 1]
    Z = img[:, :, 2]
    return X, Y, Z

def plot3d(img, cor):
    m, n = np.shape(img)
    x = np.arange(m)
    y = np.arange(n)
    Y, X = np.meshgrid(y, x)
    Z = np.zeros([m, n])
    Z[:, :] = img

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Sistema de Cor' + cor)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylabel('intensidade')
    plt.savefig('Plot/' + cor)
    plt.clf()

def plotimg(img,b):
    plt.figure()
    plt.imshow(img, cmap='gray')

def plothist(img, b):
    plt.figure()
    plt.hist(img.ravel(), np.arange(0, 256), [0, 256-1], color = 'royalblue', ec = 'k', density = 'true')
    plt.xlabel('Valor')
    plt.ylabel('Frequência')
    plt.savefig('Plot/' + b + '.jpg')
    plt.clf()


def equalhist(img):
    equ = cv2.equalizeHist(img)
    res = np.hstack((img, equ)) #stacking images side-by-side
    cv2.imwrite('res.png', res)

