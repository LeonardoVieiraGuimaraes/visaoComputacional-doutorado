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
import numpy as np
import sklearn.cluster as km
import os
from matplotlib import pyplot as plt

#Italia = 0
#RedGlobe = 1
#Vitoria = 2

pontos = [100, 200, 300]
cores = ['', 'RGB', 'Lab']
cv1 = [5]
centro = []

for points in pontos:

    extend = False
    upright = False
    octaves = 4
    octavesLayers = 2
    if extend == False:
        v = 64
    else:
        v = 128

    surf = cv2.xfeatures2d.SURF_create(points, octaves, octavesLayers, extend, upright)

    for past in cores:
        print('Pontos ' + str(points) + ' Cor ' + past)
        cx = dict()
        cy = dict()
        n = 0
        for cont, especie in enumerate(['Italia', 'RedGlobe','Vitoria']):
            for temp, tempo in enumerate(range(0,25,6)):
                for i in range(1,31):

                    #print('Especie ' + especie + ' Tempo ' + str(tempo) + ' Imagem ' + str(i))
                    #print('LBP Treino ' + ltreino)
                    pastaorigem = 'C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/' + especie + past + '/' + especie + '_' + str(tempo) + '_' + str(i) + '.jpg'
                    img = cv2.imread(pastaorigem)
                    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    d = dict()
                    kk = list()
                    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    for j in range(3):
                        k1, p1 = surf.detectAndCompute(img[:,:,j], None)
                        if p1 is None:
                            p1 = np.zeros([1,v])

                        kk = kk + k1
                        d[j] = p1

                    #imgrgb = cv2.imread('C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/' + especie + 'RGB' + '/' + especie + '_' + str(tempo) + '_' + str(i) + '.jpg')
                    #imgsurf = cv2.drawKeypoints(imgrgb,kk,None,(255,0,0),4)
                    #try:
                    #    os.makedirs('C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/SURF' + str(points) + '/')
                    #except FileExistsError:
                    #    pass

                    #cv2.imwrite('C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/SURF' + str(points) + '/' + especie + '_' + str(tempo) + '_' + str(i) + '.jpg', imgsurf)

                    cx[n] = np.concatenate((d[0],d[1],d[2]), axis = 0)
                    cy[n] = [cont, temp]
                    n = n + 1
                    ct = 0
                    for nn in range(len(cx)):
                        ct = ct + len(cx[nn])

                    arq = open('Resultados/SURF_' + str(points) + '.txt', 'w')
                    arq.write('Pontos ' + str(ct))

        for cv in cv1:
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

            for centros in centro:

                kmeans = km.KMeans(n_clusters = centros, init = 'random',  max_iter = 20, precompute_distances = False, n_jobs = -1)

                cote = -1
                cotr = -1
                for treino in range(0,cv):
                    #print('Treino ' + str(treino))
                    Xxtreino = []
                    Xxteste = []
                    ptr = Xtreino[treino]
                    pte = Xteste[treino]
                    tk = np.concatenate([ptr[j] for j in range(len(ptr))])
                    tke = np.concatenate([pte[j] for j in range(len(pte))])
                    print('Agrupamento Kmeans Treino ' + str(len(tk)) + ' ' + str(treino))
                    print('Agrupamento Kmeans Teste ' + str(len(tke)))


                    kmeans.fit(np.array(tk))

                    print('Predicao Treino ')

                    for tem in range(len(ptr)):
                        pr = kmeans.predict(ptr[tem])
                        h1 = np.histogram(pr, bins = centros, density = True)
                        Xxtreino = Xxtreino + [h1[0]]

                        if treino == 0:
                            cotr = cotr + 1
                            try:
                                os.makedirs('C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/HistSURF' + str(points) + '/' + str(centros) + '/')
                            except FileExistsError:
                                pass
                            plt.clf()
                            plt.bar(np.arange(len(h1[0])), h1[0])
                            plt.title('Bag of Featrures - SURF')
                            plt.xlabel('Valor')
                            plt.ylabel('Frequência')
                            plt.savefig('C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/HistSURF' + str(points) + '/' + str(centros) + '/trhistsurf_' + str(cotr) + '.jpg')
                            #plt.cla()
                            plt.clf()

                    print('Predicao Teste')
                    for tem in range(len(pte)):
                        pr = kmeans.predict(pte[tem])
                        h1 = np.histogram(pr, bins = centros, density = True)
                        Xxteste = Xxteste + [h1[0]]

                        if treino == 0:
                            cote = cote + 1
                            plt.clf()
                            plt.bar(np.arange(len(h1[0])), h1[0])
                            plt.title('Bag of Featrures - SURF')
                            plt.xlabel('Valor')
                            plt.ylabel('Frequência')
                            plt.savefig('C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/HistSURF' + str(points) + '/' + str(centros) +  '/tehistsurf_' + str(cote) + '.jpg')
                            plt.clf()

                    patr = 'C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/TreinoF/Treino' + str(treino) + '/'
                    pate = 'C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/TreinoF/Teste' + str(treino) + '/'
                    try:
                        os.makedirs(patr)
                    except FileExistsError:
                        pass

                    try:
                        os.makedirs(pate)
                    except FileExistsError:
                        pass

                    np.savetxt(patr + 'XtreinoSURF_' + str(points) + '_' + str(centros) + '_' + str(cv) + '.txt', np.array(Xxtreino), delimiter = " ")
                    np.savetxt(patr + 'YtreinoSURF_' + str(points) + '_' + str(centros) + '_' + str(cv) +  '.txt', np.array(Ytreino[treino]), delimiter = " ")
                    np.savetxt(pate + 'XtesteSURF_' + str(points) + '_' + str(centros) + '_' + str(cv) +  '.txt', np.array(Xxteste), delimiter = " ")
                    np.savetxt(pate + 'YtesteSURF_' + str(points) + '_' + str(centros) + '_' + str(cv) +  '.txt', np.array(Yteste[treino]), delimiter = " ")

                print('Salvo')
plt.close()

# =============================================================================
# pasta = 'C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/Fotos Validação/'
#
# Xval = []
# Yval = []
# for cont, tipo in enumerate(['Verde','Vermelho']):
#     print('Validacao ' + tipo)
#     lista = os.listdir(pasta+tipo)
#     val = []
#     for l in lista:
#         img = plt.imread(pasta + tipo + '/' + l)
#         dd = dict()
#         for j in range(3):
#             k1, p1 = surf.detectAndCompute(img[:,:,j], None)
#             if p1 is None:
#                 p1 = np.zeros([1,v])
#
#             dd[j] = p1
#
#         cc = np.concatenate((dd[0],dd[1],dd[2]), axis = 0)
#
#
#         pr = kmeans.predict(cc)
#         h1 = np.histogram(pr, bins = centros, density = True)
#
#         Xval = Xval + [h1[0]]
#         Yval = Yval + [cont]
#
# np.savetxt(pasta + 'XvalidacaoSURF_' + str(teste) + '.txt', np.array(Xval), delimiter = " ")
# np.savetxt(pasta + 'YvalidacaoSURF_' + str(teste) + '.txt', np.array(Yval), delimiter = " ")
# print('Salvo')
#
#
#
#
#
#
#
#
# =============================================================================
