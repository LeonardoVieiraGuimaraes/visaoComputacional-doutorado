# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:03:00 2019

@author: Leonardo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:30:02 2019

@author: Leonardo
"""

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

#Italia = 0
#RedGlobe = 1
#Vitoria = 2



for past in ['RGB', 'Lab']:
    n = 0
    cx = dict()
    cy = dict()
    Xtreino = dict()
    Ytreino = dict()
    Xteste = dict()
    Yteste = dict()
    for cont, especie in enumerate(['Italia', 'RedGlobe','Vitoria']):
        for temp, tempo in enumerate(range(0,25,6)):
            for i in range(1,31):
                #print('Especie ' + especie + ' Tempo ' + str(tempo) + ' Imagem ' + str(i))
                #print('LBP Treino ' + ltreino)
                pastaorigem = 'C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/' + especie + past + '/' + especie + '_' + str(tempo) + '_' + str(i) + '.jpg'
                img = cv2.imread(pastaorigem)
                #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                h1 = dict()
                for j in range(3):
                    (h1[j], _) = np.histogram(img[:,:,j].ravel(), 256,[0,256], density = True)


                cx[n] = np.concatenate((h1[0],h1[1],h1[2]), axis = None)
                cy[n] = [cont, temp]

                try:
                    os.makedirs('C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/Hist' + past + '/')
                except FileExistsError:
                    pass
                plt.clf()
                plt.bar(np.arange(len(cx[n])), cx[n])
                plt.title('BGR')
                plt.xlabel('Valor')
                plt.ylabel('Frequência')
                plt.savefig('C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/Hist' + past + '/' + especie + '_' + str(tempo) + '_' + str(i) + '.jpg')
                plt.clf()
                n = n+1

        print('Termino da extracao ' + especie)


    for cv in [3, 5, 6]:
        print('Iniciando cv ' + str(cv) )

        Xtreino = dict()
        Ytreino = dict()
        Xteste = dict()
        Yteste = dict()
        for treino in range(0,cv):
            Xtreino[treino] = list()
            Ytreino[treino] = list()
            Xteste[treino] = list()
            Yteste[treino] = list()

        i = 0
        for cc in range(len(cx)):
            for treino in range(0,cv):

                if ((i) % cv) == treino:
                    Xteste[treino] = Xteste[treino] + [cx[cc]]
                    Yteste[treino] =Yteste[treino] + [cy[cc]]

                else:
                    Xtreino[treino] = Xtreino[treino] + [cx[cc]]
                    Ytreino[treino] = Ytreino[treino] + [cy[cc]]
            i = i + 1
            if i == 30:
                i = 0

        for treino in range(0,cv):
            ptr = 'C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/TreinoF/Treino' + str(treino) + '/'
            pte = 'C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/TreinoF/Teste' + str(treino) + '/'
            try:
                os.makedirs(ptr)
            except FileExistsError:
                pass

            try:
                os.makedirs(pte)
            except FileExistsError:
                pass

            np.savetxt(ptr + 'Xtreino' + past + '_' + str(cv) + '.txt', np.array(Xtreino[treino]), delimiter = " ")
            np.savetxt(ptr + 'Ytreino' + past + '_' + str(cv) + '.txt', np.array(Ytreino[treino]), delimiter = " ")
            np.savetxt(pte + 'Xteste' + past + '_' + str(cv) + '.txt', np.array(Xteste[treino]), delimiter = " ")
            np.savetxt(pte + 'Yteste' + past + '_' + str(cv) + '.txt', np.array(Yteste[treino]), delimiter = " ")

        print('Salvo')

    plt.close()
    # =============================================================================
    # pasta = 'C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/Fotos Validação/'
    #
    # Xval = []
    # Yval = []
    #
    # for cont, tipo in enumerate(['Verde','Vermelho']):
    #     lista = os.listdir(pasta+tipo)
    #     val = []
    #     for l in lista:
    #         img = plt.imread(pasta + tipo + '/' + l)
    #
    #         cc = []
    #         for j in range(3):
    #             h1 = np.histogram(img[:,:,j].ravel(),256,[0,256], density = True)
    #             cc.extend(h1[0])
    #
    #         Xval = Xval + [cc]
    #         Yval = Yval + [cont]
    #
    # np.savetxt(pasta + 'XvalidacaoRGB_' + str(teste) + '.txt', np.array(Xval), delimiter = " ")
    # np.savetxt(pasta + 'YvalidacaoRGB_' + str(teste) + '.txt', np.array(Yval), delimiter = " ")
    #
    #
    #
    #
    #
    #
    # =============================================================================
