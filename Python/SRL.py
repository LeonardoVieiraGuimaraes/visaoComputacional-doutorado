# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 00:49:30 2019

@author: Leonardo
"""

import numpy as np
#from sklearn.neural_network import MLPClassifier
#from sklearn import metrics
import os

Xtreino = dict()
Ytreino = dict()
Xteste = dict()
Yteste = dict()


cores = ['RGB', 'Lab']
pontos = [100]
centro = [50]
cv1 = [5]

for past in cores:
    for points in pontos:
        for centros in centro:
            for cv in cv1:
                for treino in range(0,cv):

                    patr = 'C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/TreinoF/Treino' + str(treino) + '/'
                    pate = 'C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/TreinoF/Teste' + str(treino) + '/'


                    for cont, tipo in enumerate(['SURF_' + str(points) + '_' + str(centros) + '_' + str(cv), past + '_' + str(cv)]):

                        Xtreino[cont] = np.loadtxt(patr + 'Xtreino' + tipo + '.txt')
                        Ytreino = np.loadtxt(patr + 'Ytreino' + tipo + '.txt')

                        Xteste[cont] = np.loadtxt(pate + 'Xteste' + tipo + '.txt')
                        Yteste = np.loadtxt(pate + 'Yteste' + tipo + '.txt')



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

                    np.savetxt(ptr + 'XtreinoSURF' + past + '_'  + str(points) + '_' + str(centros) + '_' + str(cv) + '.txt', np.concatenate((Xtreino[0],Xtreino[1]), axis = 1), delimiter = " ")
                    np.savetxt(ptr + 'YtreinoSURF' + past + '_'  + str(points) + '_' + str(centros) + '_' + str(cv) + '.txt', Ytreino, delimiter = " ")
                    np.savetxt(pte + 'XtesteSURF' + past + '_'  + str(points) + '_' + str(centros) + '_' + str(cv) + '.txt', np.concatenate((Xteste[0],Xteste[1]), axis = 1), delimiter = " ")
                    np.savetxt(pte + 'YtesteSURF' + past + '_'  + str(points) + '_' + str(centros) + '_' + str(cv) + '.txt', Yteste, delimiter = " ")

                print('Salvo_' + str(points) + '_' + str(centros) + '_' + str(cv))



    # =============================================================================
    # pasta = 'C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/Fotos Validação/'

    # Xval = dict()
    # Yval = dict()
    # for cont, tipo in enumerate(['LBP', 'SURF']):
    #
    #     Xval[cont] = np.loadtxt(pasta + 'Xvalidacao' + tipo + '.txt', delimiter = " ")
    #     Yval[cont] = np.loadtxt(pasta + 'Yvalidacao' + tipo + '.txt', delimiter = " ")
    #
    # np.savetxt(pasta + 'XvalidacaoLBPSURF.txt',np.concatenate((Xval[0],Xval[1]), axis = 1), delimiter = " ")
    # np.savetxt(pasta + 'YvalidacaoLBPSURF.txt', Yval[0], delimiter = " ")
    #
    #
    # =============================================================================
