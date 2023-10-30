# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 20:28:22 2019

@author: Leonardo
"""

import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

def cinza(img):
    r = 0.5
    g = 0.5
    b = 0
    ec = b*img[:,:,2] +  g*img[:,:,1] + r*img[:,:,0]


    return np.uint8(np.around(ec))

def retirad(imagem,r,g,b):
        if r == 0:
            imagem[:,:,2] = 0    #elimina o vermelho
        if g == 0:
            imagem[:,:,1] = 0    #elimina o verde
        if b == 0:
            imagem[:,:,0] = 0    #elimina o azul
        return imagem

for especie in ['Italia', 'RedGlobe', 'Vitoria']:

    for tempo in range(0,25,6):

        tempo = str(tempo)

        imgrgb = cv2.imread('C:/Users/Leonardo/Google Drive/Imagens uva/Uvas/' + especie + '/' + tempo + '.jpg')

        filtro = cv2.GaussianBlur(imgrgb,(101,101),0)

        imgLab = cv2.cvtColor(filtro,cv2.COLOR_BGR2Lab)



        gray = cv2.cvtColor(imgLab,cv2.COLOR_RGB2GRAY)

        #gray = cv2.GaussianBlur(gray,(101,101),0)

        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY +cv2.THRESH_OTSU)

        # noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        #kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 5)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=12)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.45*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers1 = markers+1

        # Now, mark the region of unknown with zero
        markers1[unknown==255] = 0

        markers2 = cv2.watershed(imgrgb,markers1)

        pasta = "Fotos/Processo" + especie
        try:
            os.makedirs(pasta)
        except FileExistsError:
            pass


        foto = especie + '_' + tempo

        cv2.imwrite(pasta + '/' + foto + '_imgLab.jpg' ,imgLab)

        cv2.imwrite(pasta + '/' + foto + '_filtro.jpg',filtro)

        cv2.imwrite(pasta + '/' + foto + '_RGB.jpg',imgrgb)

        cv2.imwrite(pasta + '/' + foto + '_ECinza.jpg',gray)

        cv2.imwrite(pasta + '/' + foto + '_Segmentacao.jpg',thresh)

        cv2.imwrite(pasta + '/' + foto + '_Abertura.jpg',opening)

        cv2.imwrite(pasta + '/' + foto + '_Dilatacao.jpg',sure_bg)

        cv2.imwrite(pasta + '/' + foto + '_TDistancia.jpg',dist_transform)

        cv2.imwrite(pasta + '/' + foto + '_TDistanciafg.jpg',sure_fg)

        cv2.imwrite(pasta + '/' + foto + '_RDesconhecida.jpg',unknown)

        cv2.imwrite(pasta + '/' + foto + 'Llink.jpg',markers)

        cv2.imwrite(pasta + '/' + foto + '_RDesconhecida0.jpg',markers1)

        #cv2.imwrite(pasta + '/' + foto + '_RDesconhecidaW.jpg',markers2)

        plt.imsave(pasta + '/' + foto + '_RDesconhecidaW2.jpg',markers2, cmap = 'jet')


        for label in np.unique(markers2):
            # if the label is zero, we are examining the 'background'
            # so simply ignore it
            if label == 1:
                continue

            if label == -1:
                continue
            #if label == -1:
            #    continue
            mask = np.zeros(imgrgb.shape, dtype="uint8")
            if especie == 'Vitoria':
                imgrgb = cv2.imread('C:/Users/Leonardo/Google Drive/Imagens uva/Uvas/' + especie + '/original/' + tempo + '.jpg')
                imgLab = cv2.cvtColor(imgrgb,cv2.COLOR_BGR2Lab)

            # otherwise, allocate memory for the label region and draw
            # it on the mask


            mask[markers2 == label] = 255

            imgR = cv2.bitwise_and(mask,imgrgb)
            imgL = cv2.bitwise_and(mask,imgLab)

            mask = np.uint8(mask)
            im2, contours, hierarchy = cv2.findContours(cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY),1,2)
            cnt = contours[0]

            rect = cv2.minAreaRect(cnt)
            x,y,w,h = cv2.boundingRect(cnt)


            N = imgR[round(y):round(y+h), round(x):round(x+w), :]
            N = np.uint8(N)

            N1 = imgL[round(y):round(y+h), round(x):round(x+w), :]
            N1 = np.uint8(N1)

            N2 = imgrgb[round(y):round(y+h), round(x):round(x+w), :]
            N2 = np.uint8(N2)

            N3 = imgLab[round(y):round(y+h), round(x):round(x+w), :]
            N3 = np.uint8(N3)


            foto = especie + '_' + tempo + '_' + str(label-1) + ".jpg"

            pasta = "Fotos/" + especie + 'R'
            try:
                os.makedirs(pasta)
            except FileExistsError:
                pass
            cv2.imwrite(pasta + '/' + foto,N)

            pasta = "Fotos/" + especie + 'L'
            try:
                os.makedirs(pasta)
            except FileExistsError:
                pass
            cv2.imwrite(pasta + '/' + foto,N1)

            pasta = "Fotos/" + especie + 'RGB'
            try:
                os.makedirs(pasta)
            except FileExistsError:
                pass
            cv2.imwrite(pasta + '/' + foto,N2)

            pasta = "Fotos/" + especie + 'Lab'
            try:
                os.makedirs(pasta)
            except FileExistsError:
                pass
            cv2.imwrite(pasta + '/' + foto,N3)

        print(tempo + " Teminado")
    print(especie + " Terminado")


