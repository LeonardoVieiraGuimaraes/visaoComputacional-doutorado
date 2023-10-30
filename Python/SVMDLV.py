# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:47:26 2019

@author: Leonardo
"""

from sklearn import svm
import numpy as np
#from sklearn.metrics import classification_report, confusion_matrix
#from sklearn import metrics
#Italia = 0
#RedGlobe = 1
#Vitoria = 2

def MatrixC(Ytes, Ypre,n):
    M = np.zeros([n,n])
    for k in range(len(Ytes)):
        M[int(Ytes[k]),int(Ypre[k])] = M[int(Ytes[k]),int(Ypre[k])  ] + 1
    return M

clf1 = svm.SVC(gamma='scale', decision_function_shape='sigmoid', random_state=10, C = 100)
clf2 = svm.SVC(gamma='scale', decision_function_shape='sigmoid', random_state=10, C = 100)

cores = ['', 'RGB', 'Lab']
raios = [2]
cv1 = [5]

for past in cores:
    for radius in raios:
        for cv in cv1:
            arqm = open('C:/Users/Leonardo/Desktop/dissertacao/dissertacao/Resultados/SVM/LBPRGBLabE' + '_' + str(cv) + '_' + str(radius) + '.tex', 'w')
            for e, esp in enumerate(['Verde', 'Vermelho']):
                arqt = open('C:/Users/Leonardo/Desktop/dissertacao/dissertacao/Resultados/SVM/LBPRGBLabT' + '_' + str(cv) + '_' + str(radius) + '_' +  esp + '.tex', 'w')


for past in cores:
    for radius in raios:
        for cv in cv1:

            Me = np.zeros([3,3])
            Mt = np.zeros([5,5,3])
            Mr = np.zeros([5,5,3])

            for i in range(0,cv):
                patr = 'C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/TreinoF/Treino' + str(i) + '/'
                pate = 'C:/Users/Leonardo/Google Drive/Imagens uva/Fotos/TreinoF/Teste' + str(i) + '/'

                Xtreino = np.loadtxt(patr + 'XtreinoLBP' + past + '_' + str(radius) + '_' + str(cv) + '.txt')
                Ytreino = np.loadtxt(patr + 'YtreinoLBP' + past + '_' + str(radius) + '_' + str(cv) + '.txt')

                Xteste = np.loadtxt(pate + 'XtesteLBP' + past + '_' + str(radius) + '_' + str(cv) + '.txt')
                Yteste = np.loadtxt(pate + 'YtesteLBP' + past + '_' + str(radius) + '_' + str(cv) + '.txt')

                print('Inicando Rede SVM ' + str(i))
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

            arq = open('C:/Users/Leonardo/Google Drive/Dissertação Final/texto/Resultados/SVM/LBP' + past + '_'  + str(radius) + '_' + str(cv) + '.tex', 'w')



            arq.write('\\begin{table}[!htb] \n\\centering \n\\begin{tabular}{|l|l|r|r|r||r|r|r|} \n\\cline{3-8} \n\\multicolumn{2}{c|}{} & \\multicolumn{3}{c||}{Predição} & \\multicolumn{3}{c|}{Resultados} \\\ \\cline{3-8}\n\\multicolumn{2}{l|}{}& Italia & RedGlobe & Vitoria & Precisão & Sensibilidade&Especificidade\\\\ \\cline{1-8} \n\multirow{3}{*}{\\rotatebox[origin=c]{90}{Real}}')

            for e, esp in enumerate(['Italia', 'RedGlobe','Vitoria']):

                arq.write('&' + esp + ' ')

                for cont, especie in enumerate(['Italia', 'RedGlobe','Vitoria']):

                    arq.write(' & ' + str(np.around(Me[e,cont], decimals = 1)))

                arq.write(' & ' + str(np.around(ACC2[e], decimals = 2)))
                arq.write(' & ' + str(np.around(TPR2[e], decimals = 2)))
                arq.write(' & ' + str(np.around(TNR2[e], decimals = 2)))
                if e == 2:
                    arq.write('\\\\ \n\\cline{1-8} \n')
                else:
                    arq.write('\\\\ \\cline{2-8} \n')


            arq.write('\\end{tabular} \n\\caption{Matriz de confusão LBP ' + past + ' com SVM} \n\label{tab:' + past+ '} \n\\end{table}')

            arq.write('\n\n')
            arq.write('Acuracia Especie LBP ' + past + ' SVM = ' + str(np.around(ACG2, decimals = 2)))
            arq.write('\n\n')

            for e, esp in enumerate(['Italia', 'RedGlobe','Vitoria']):

                arq.write('\\begin{table}[!htb] \n\\centering \n\\begin{tabular}{|l|l|r|r|r|r|r||r|r|r|} \n\\cline{3-10} \n\\multicolumn{2}{c|}{} & \\multicolumn{5}{c||}{Predição} & \\multicolumn{3}{c|}{Resultados} \\\ \\cline{3-10}\n\\multicolumn{2}{l|}{}& 0 & 6 & 12 & 18 & 24 & Precisão & Sensibilidade & Especificidade\\\\ \\cline{1-10} \n\multirow{5}{*}{\\rotatebox[origin=c]{90}{Real}}')

                for t, tempo in enumerate(['0','6','12','18','24']):

                    arq.write('&' + tempo + ' ')

                    for cont, temp in enumerate(['0','6','12','18','24']):

                        arq.write(' & ' + str(np.around(M1[e][t,cont], decimals = 1)))

                    arq.write(' & ' + str(np.around(ACC1[e][t], decimals = 2)))
                    arq.write(' & ' + str(np.around(TPR1[e][t], decimals = 2)))
                    arq.write(' & ' + str(np.around(TNR1[e][t], decimals = 2)))
                    if t == 4:
                        arq.write('\\\\ \n\\cline{1-10} \n')
                    else:
                        arq.write('\\\\ \\cline{2-10} \n')

                arq.write('\\end{tabular} \n\\caption{Matriz de confusão LBP ' + past + ' com SVM tipo ' + esp + '} \n\label{tab:SVMLBP' + past + '} \n\\end{table}')

                arq.write('\n\n')
                arq.write('Acuracia LBP ' + esp + ' ' + past + ' SVM = ' + str(np.around(ACG1[e],decimals = 2)))
                arq.write('\n\n')


            for e, esp in enumerate(['Italia', 'RedGlobe','Vitoria']):

                arq.write('\\begin{table}[!htb] \n\\centering \n\\begin{tabular}{|l|l|r|r|r|r|r||r|r|r|} \n\\cline{3-10} \n\\multicolumn{2}{c|}{} & \\multicolumn{5}{c||}{Predição} & \\multicolumn{3}{c|}{Resultados} \\\ \\cline{3-10}\n\\multicolumn{2}{l|}{}& 0 &  6 & 12 & 18 & 24 & Precisão & Sensibilidade & Especificidade\\\\ \\cline{1-10} \n\multirow{5}{*}{\\rotatebox[origin=c]{90}{Real}}')

                for t, tempo in enumerate(['0','6','12','18','24']):

                    arq.write('&' + tempo + ' ')

                    for cont, temp in enumerate(['0','6','12','18','24']):

                        arq.write(' & ' + str(np.around(M[e][t,cont], decimals = 1)))

                    arq.write(' & ' + str(np.around(ACC[e][t], decimals = 2)))
                    arq.write(' & ' + str(np.around(TPR[e][t], decimals = 2)))
                    arq.write(' & ' + str(np.around(TNR[e][t], decimals = 2)))
                    if t == 4:
                        arq.write('\\\\ \n\\cline{1-10} \n')
                    else:
                        arq.write('\\\\ \\cline{2-10} \n')

                arq.write('\\end{tabular} \n\\caption{Matriz de confusão LBP ' + past + ' com SVM tipo ' + esp + ' /Sistema} \n\label{tab:SVMsLBP' + past+ '} \n\\end{table}')

                arq.write('\n\n')
                arq.write('Acuracia LBP ' + esp + ' ' + past + ' SVM = ' + str(np.around(ACG[e],decimals = 2)))
                arq.write('\n\n')

            arq.close()

        arqm = open('C:/Users/Leonardo/Desktop/dissertacao/dissertacao/Resultados/SVM/LBPRGBLabE' + '_' + str(cv) + '_' + str(radius) + '.tex', 'a+')

        arqm.write('LBP ' + str(radius) + ' ' + past +  '     &     ' + str(np.around(PR2[0],decimals = 2))  + '     &    ' +  str(np.around(PR2[1],decimals = 2))  + '    &    ' + str(np.around(PR2[2], decimals = 2))  + '     &   ' + str(np.around(TPR2[0],decimals = 2))  + '     &     ' + str(np.around(TPR2[1],decimals = 2))  + '     &    ' + str(np.around(TPR2[2],decimals = 2))  + '     &    ' + str(np.around(TNR2[0],decimals = 2))  + '    &       ' + str(np.around(TNR2[1], decimals = 2))  + '   &   ' + str(np.around(TNR2[2],decimals=2))  + '   &   '  + str(np.around(ACG2,decimals = 2)) +  ' \\\ \n')

        for e, esp in enumerate(['Italia', 'RedGlobe','Vitoria']):

            arqt = open('C:/Users/Leonardo/Desktop/dissertacao/dissertacao/Resultados/SVM/LBPRGBLabT' + '_' + str(cv) + '_' + str(radius) + '_' +  esp + '.tex', 'a+')

            arqt.write('LBP ' + str(radius) + ' ' + past + '     &     ' + str(np.around(PR1[e][0],decimals = 2))  + '     &    ' +  str(np.around(PR1[e][1],decimals = 2))  + '    &    ' + str(np.around(PR1[e][2], decimals = 2))  + '     &   '  + str(np.around(PR1[e][3],decimals = 2))  + '     &    '  + str(np.around(PR1[e][4],decimals = 2))  + '     &    ' + str(np.around(TPR1[e][0],decimals = 2))  + '     &     ' + str(np.around(TPR1[e][1],decimals = 2))  + '     &    ' + str(np.around(TPR1[e][2],decimals = 2))  + '     &    ' + str(np.around(TPR1[e][3],decimals = 2))  + '     &    ' + str(np.around(TPR1[e][4],decimals = 2))  + '     &    ' +  str(np.around(TNR1[e][0],decimals = 2))  + '    &       ' + str(np.around(TNR1[e][1], decimals = 2))  + '   &   ' + str(np.around(TNR1[e][2],decimals=2))  + '   &   ' + str(np.around(TNR1[e][3],decimals = 2))  + '    &       ' + str(np.around(TNR1[e][4],decimals = 2))  + '   &   '  + str(np.around(ACG[e],decimals = 2)) + ' \\\ \n')

        print('Saldo')


#arqm.write('\\hline \n \\end{tabular} \n \\end{table}')


arqt.close()

arqm.close()
