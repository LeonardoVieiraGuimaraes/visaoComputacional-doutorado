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
from skimage.feature import local_binary_pattern
from matplotlib import pyplot as plt

#Italia = 0
#RedGlobe = 1
#Vitoria = 2
cores = ['', 'RGB', 'Lab']
raio = [2,4]
cv1 = [5]


for radius in raio:
    npoints = 8 * radius
    #teste = 1
    n = 0
    for past in cores:
        print('Radius ' + str(radius) + ' Cor ' + past)
        cx = dict()
        cy = dict()
        nn = 0
        for cont, especie in enumerate(['Italia', 'RedGlobe','Vitoria']):
            for temp, tempo in enumerate(range(0,25,6)):
                for i in range(1,31):
                    #print('Especie ' + especie + ' Tempo ' + str(tempo) + ' Imagem ' + str(i))
                    #print('LBP Treino ' + ltreino)
                    pastaorigem = 'C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/' + especie + past + '/' + especie + '_' + str(tempo) + '_' + str(i) + '.jpg'
                    img = cv2.imread(pastaorigem)
                    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    h1 = dict()
                    l,ll,lll = img.shape
                    pp = np.zeros([l,ll], dtype="uint8")
                    for j in range(3):
                        p1 =  local_binary_pattern(img[:,:,j], npoints, radius, method='uniform')
                        n_bins = int(p1.max() + 1)
                        (h1[j], _) = np.histogram(p1.ravel(), bins = n_bins , range=(0, n_bins), density = True)
                        pp = pp + p1

                    try:
                        os.makedirs('C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/LBP' + str(radius) + '/')
                    except FileExistsError:
                        pass

                    #plt.imsave('C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/LBP' + str(radius) + '/' + especie + '_' + str(tempo) + '_' + str(i) + '.jpg', np.uint8(pp/3), cmap='gray')

                    cx[nn] = np.concatenate((h1[0],h1[1],h1[2]), axis = None)
                    cy[nn] = [cont, temp]
                    try:
                        os.makedirs('C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/HistLBP' + str(radius) + '/')
                    except FileExistsError:
                        pass
                    plt.clf()
                    plt.bar(np.arange(len(cx[nn])), cx[nn])
                    plt.title('LBP')
                    plt.xlabel('Valor')
                    plt.ylabel('Frequência')
                    plt.savefig('C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/HistLBP' + str(radius) + '/histlbp_' + str(n) + '.jpg')
                    plt.cla()
                    plt.clf()
                    n = n + 1
                    nn = nn + 1

        for cv in cv1:
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

                np.savetxt(ptr + 'XtreinoLBP_' + str(radius) + '_' + str(cv) + '.txt', np.array(Xtreino[treino]), delimiter = " ")
                np.savetxt(ptr + 'YtreinoLBP_' + str(radius) + '_' + str(cv) + '.txt', np.array(Ytreino[treino]), delimiter = " ")
                np.savetxt(pte + 'XtesteLBP_'  + str(radius) + '_' + str(cv) +  '.txt', np.array(Xteste[treino]), delimiter = " ")
                np.savetxt(pte + 'YtesteLBP_'  + str(radius) + '_' + str(cv) +  '.txt', np.array(Yteste[treino]), delimiter = " ")

            print('Salvo ' + str(cv))
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
    #         h1 = dict()
    #         for j in range(3):
    #             p1 =  local_binary_pattern(img[:,:,j], npoints, radius, method='uniform')
    #             (h1[j],_) = np.histogram(p1.ravel(), bins = b, range = it, density = True)
    #
    #         cc = np.concatenate((h1[0],h1[1],h1[2]), axis = None)
    #
    #         Xval = Xval + [cc]
    #         Yval = Yval + [cont]
    #
    # np.savetxt(pasta + 'XvalidacaoLBP_' + str(teste) + '.txt', np.array(Xval), delimiter = " ")
    # np.savetxt(pasta + 'YvalidacaoLBP_' + str(teste) + '.txt', np.array(Yval), delimiter = " ")
    #
    #
    #
    #
    #
    #
    # =============================================================================
