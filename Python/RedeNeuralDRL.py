# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 08:55:09 2019

@author: Leonardo
"""
import numpy as np
from sklearn.neural_network import MLPClassifier

#import csv
#from sklearn import metrics
#import scikitplot as skplt
#import matplotlib.pyplot as plt
#from sklearn.model_selection import cross_val_predict, cross_val_score


def MatrixC(Ytes, Ypre,n):
    M = np.zeros([n,n])
    for k in range(len(Ytes)):
        M[int(Ytes[k]),int(Ypre[k])] = M[int(Ytes[k]),int(Ypre[k])  ] + 1
    return M

clf1 = MLPClassifier(solver = 'lbfgs', alpha = 1e-20, hidden_layer_sizes=(100, 100), random_state=10, activation = 'tanh', max_iter = 100)
clf2 = MLPClassifier(solver = 'lbfgs', alpha = 1e-20, hidden_layer_sizes=(100, 100), random_state=10, activation = 'tanh', max_iter = 100)

cores = ['RGB', 'Lab']
cv1 = [5]

for past in cores:
    for cv in cv1:
        arqm = open('C:/Users/Leonardo/Desktop/Dissertacao/Resultados/MLP/RGBLabE' + '_' + str(cv) + '.tex', 'w')
        for e, esp in enumerate(['Italia', 'RedGlobe','Vitoria']):
            arqtp = open('C:/Users/Leonardo/Desktop/Dissertacao/Resultados/MLP/RGBLabT_P' + '_' + str(cv) + '_' +  esp + '.tex', 'w')

            arqtse = open('C:/Users/Leonardo/Desktop/Dissertacao/Resultados/MLP/RGBLabT_SE' + '_' + str(cv) + '_' +  esp + '.tex', 'w')

            arqta = open('C:/Users/Leonardo/Desktop/Dissertacao/Resultados/MLP/RGBLabT_A' + '_' + str(cv) + '_' +  esp + '.tex', 'w')
            arqme = open('C:/Users/Leonardo/Desktop/Dissertacao/Resultados/MLP/RGBLabT_ME' + '_' + str(cv) + '_' +  esp + '.tex', 'w')


for past in cores:
    for cv in cv1:

        Me = np.zeros([3,3])
        Mt = np.zeros([5,5,3])
        Mr = np.zeros([5,5,3])

        for i in range(0,cv):
            patr = 'C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/TreinoF/Treino' + str(i) + '/'
            pate = 'C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/TreinoF/Teste' + str(i) + '/'

            Xtreino = np.loadtxt(patr + 'Xtreino' + past + '_' + str(cv) + '.txt')
            Ytreino = np.loadtxt(patr + 'Ytreino' + past + '_' + str(cv) + '.txt')

            Xteste = np.loadtxt(pate + 'Xteste' + past + '_'  + str(cv) + '.txt')
            Yteste = np.loadtxt(pate + 'Yteste' + past + '_' + str(cv) +  '.txt')

            print('Inicando Rede MLP ' + str(i))
            clf1.fit(Xtreino, Ytreino[:,0])
            y_pred1 = clf1.predict(Xteste)

            Me = Me + MatrixC(Yteste[:,0], y_pred1, 3)

            XItalia = []
            XRedGlobe = []
            XVitoria = []
            YItalia = []
            YRedGlobe = []
            YVitoria = []

            for j, e in enumerate(y_pred1):
                e = int(e)

                if e == 0 and Yteste[j,0] == 0:
                    XItalia = XItalia + [Xteste[j,:]]
                    YItalia = YItalia + [int(Yteste[j,1])]

                if e == 1 and Yteste[j,0] == 1:
                    XRedGlobe = XRedGlobe + [Xteste[j,:]]
                    YRedGlobe = YRedGlobe + [int(Yteste[j,1])]

                if e == 2 and Yteste[j,0] == 2:
                    XVitoria = XVitoria + [Xteste[j,:]]
                    YVitoria = YVitoria + [int(Yteste[j,1])]

            for cont, especie in enumerate(['Italia', 'RedGlobe','Vitoria']):

                xtreino = Xtreino[0+int(150-150/cv)*cont:int(150-150/cv)+int(150-150/cv)*cont,:]
                ytreino = Ytreino[0+int(150-150/cv)*cont:int(150-150/cv)+int(150-150/cv)*cont,1]

                xteste = Xteste[0+int(150/cv)*cont:int(150/cv)+int(150/cv)*cont,:]
                yteste = Yteste[0+int(150/cv)*cont:int(150/cv)+int(150/cv)*cont,1]

                #print('Inicando Rede Tempo ' + str(i) + ' ' + especie)
                clf2.fit(xtreino, ytreino)
                y_pred2 = clf2.predict(xteste)


                Mt[:,:,cont] = Mt[:,:,cont] + MatrixC(yteste, y_pred2,5)

                if cont == 0:
                    y_pred3 = clf2.predict(np.array(XItalia))
                    Mr[:,:,0] = Mr[:,:,0] + MatrixC(YItalia,y_pred3, 5)

                if cont == 1:
                    y_pred3 = clf2.predict(np.array(XRedGlobe))
                    Mr[:,:,1] = Mr[:,:,1] + MatrixC(YRedGlobe,y_pred3,5)

                if cont == 2:
                    y_pred3 = clf2.predict(np.array(XVitoria))
                    Mr[:,:,2] = Mr[:,:,2] + MatrixC(YVitoria,y_pred3, 5)

        Me = 100*(Me/cv)/(150/cv)

        M = list()
        M1 = list()
        for i in range(3):
            M = M + [100*(Mr[:,:,i]/cv)/(30/cv)]
            M1 = M1 + [100*(Mt[:,:,i]/cv)/(30/cv)]

        FP = []
        FN = []
        TP = []
        TN = []
        TPR = []
        TNR = []
        PR = []
        ACC = []
        ACG = []

        for p in range(len(M)):
            FP = FP + [M[p].sum(axis = 0) - np.diag(M[p])]
            FN = FN + [M[p].sum(axis = 1) - np.diag(M[p])]
            TP = TP + [np.diag(M[p])]
            TN = TN + [M[p].sum() - FP[p] - FN[p] - TP[p]]
            #sensibilidde, hit, rate, recaal, or true postivite rate
            TPR = TPR + [TP[p]/(TP[p]+FN[p])]

            #specificity ou ture negative rate
            TNR = TNR + [TN[p]/(TN[p] + FP[p])]

            #Precisão
            PR = PR + [(TP[p])/(TP[p]+FP[p])]

            #Acuracia
            ACC = ACC + [(TP[p] + TN[p])/(TP[p] + FN[p] + FP[p] + TN[p])]

            #Acurácia Geral
            ACG = ACG + [np.diag(M[p]).sum()/M[p].sum()]



        FP1 = []
        FN1 = []
        TP1 = []
        TN1 = []
        TPR1 = []
        TNR1 = []
        PR1 = []
        ACC1 = []
        ACG1 = []

        for p in range(len(M1)):
            FP1 = FP1 + [M1[p].sum(axis = 0) - np.diag(M1[p])]
            FN1 = FN1 + [M1[p].sum(axis = 1) - np.diag(M1[p])]
            TP1 = TP1 + [np.diag(M1[p])]
            TN1 = TN1 + [M1[p].sum() - FP1[p] - FN1[p] - TP1[p]]
            #sensibilidde, hit, rate, recaal, or true postivite rate
            TPR1 = TPR1 + [TP1[p]/(TP1[p]+FN1[p])]

            #specificity ou ture negative rate
            TNR1 = TNR1 + [TN1[p]/(TN1[p] + FP1[p])]

            #Precisão
            PR1 = PR1 + [(TP1[p])/(TP1[p]+FP1[p])]

            #Acuracia
            ACC1 = ACC1 + [(TP1[p] + TN1[p])/(TP1[p] + FN1[p] + FP1[p] + TN1[p])]

            #Acurácia Geral
            ACG1 = ACG1 + [np.diag(M1[p]).sum()/M1[p].sum()]

        FP2 = Me.sum(axis = 0) - np.diag(Me)
        FN2 = Me.sum(axis = 1) - np.diag(Me)
        TP2 = np.diag(Me)
        TN2 = Me.sum() - FP2 - FN2 - TP2

        #sensibilidde, hit, rate, recaal, or true postivite rate
        TPR2 = TP2/(TP2+FN2)

        #specificity ou ture negative rate
        TNR2 = TN2/(TN2 + FP2)

        #Precisão
        PR2 = (TP2)/(TP2+FP2)

        #Acuracia
        ACC2 = (TP2 + TN2)/(TP2 + FN2 + FP2 + TN2)

        #Acurácia Geral
        ACG2 = np.diag(Me).sum()/Me.sum()


        arqm = open('C:/Users/Leonardo/Desktop/Dissertacao/Resultados/MLP/RGBLabE' + '_' + str(cv) + '.tex', 'a+')

        arqm.write(past + '     &     ' + str(np.around(PR2[0],decimals = 2))  + '     &    ' +  str(np.around(PR2[1],decimals = 2))  + '    &    ' + str(np.around(PR2[2], decimals = 2))  + '     &   ' + str(np.around(TPR2[0],decimals = 2))  + '     &     ' + str(np.around(TPR2[1],decimals = 2))  + '     &    ' + str(np.around(TPR2[2],decimals = 2))  + '     &    ' + str(np.around(TNR2[0],decimals = 2))  + '    &       ' + str(np.around(TNR2[1], decimals = 2))  + '   &   ' + str(np.around(TNR2[2],decimals=2))  + '    &     ' +  str(np.around(ACG2,decimals = 2)) + ' \\\ \n')

        for e, esp in enumerate(['Italia', 'RedGlobe','Vitoria']):

            arqtp = open('C:/Users/Leonardo/Desktop/Dissertacao/Resultados/MLP/RGBLabT_P' + '_' + str(cv) + '_' +  esp + '.tex', 'a+')

            arqtse = open('C:/Users/Leonardo/Desktop/Dissertacao/Resultados/MLP/RGBLabT_SE' + '_' + str(cv) + '_' +  esp + '.tex', 'a+')

            arqta = open('C:/Users/Leonardo/Desktop/Dissertacao/Resultados/MLP/RGBLabT_A' + '_' + str(cv) + '_' +  esp + '.tex', 'a+')

            arqtp.write(past + '     &     ' + str(np.around(PR1[e][0],decimals = 2))  + '     &    ' +  str(np.around(PR1[e][1],decimals = 2))  + '    &    ' + str(np.around(PR1[e][2], decimals = 2))  + '     &   '  + str(np.around(PR1[e][3],decimals = 2))  + '     &    '  + str(np.around(PR1[e][4],decimals = 2))  + ' \\\ \n')

            arqtse.write(past + '     &     ' + str(np.around(TPR1[e][0],decimals = 2))  + '     &     ' + str(np.around(TPR1[e][1],decimals = 2))  + '     &    ' + str(np.around(TPR1[e][2],decimals = 2))  + '     &    ' + str(np.around(TPR1[e][3],decimals = 2))  + '     &    ' + str(np.around(TPR1[e][4],decimals = 2))  + '     &     ' +  str(np.around(TNR1[e][0],decimals = 2))  + '    &       ' + str(np.around(TNR1[e][1], decimals = 2))  + '   &   ' + str(np.around(TNR1[e][2],decimals=2))  + '   &   ' + str(np.around(TNR1[e][3],decimals = 2))  + '    &       ' + str(np.around(TNR1[e][4],decimals = 2))  + ' \\\ \n')

            arqta.write(past + '     &     '  + str(np.around(ACG1[e],decimals = 2)) + ' \\\ \n')

            arqme = open('C:/Users/Leonardo/Desktop/Dissertacao/Resultados/MLP/RGBLabT_ME' + '_' + str(cv) + '_' +  esp + '.tex', 'a+')
            arqme.write(past + ' Precisão ' + str(np.around(np.mean(PR1[e]),decimals = 2)) + ' Sensibilidade ' + str(np.around(np.mean(TPR1[e]),decimals = 2)) + ' Especificidade ' + str(np.around(np.mean(TNR1[e]),decimals = 2)) + ' Acurácia ' + str(np.around(ACG1[e],decimals = 2)) + '\n')



        print('Salvo')


#arqm.write('\hline \n \end{tabular} \n \end{table}')


arqtp.close()
arqtse.close()
arqta.close()
arqm.close()
arqme.close()