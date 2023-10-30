# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 16:47:53 2021

@author: leona
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import numpy as np
import cv2 as cv


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
# if __name__ == '__main__':
#      freeze_support()
     
     
transform = transforms.Compose(
    [transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 100

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


sift = cv.SIFT_create()



if __name__ == '__main__':
    dataiter = iter(trainloader)
    img_train, lab_train = dataiter.next()
    img_train = img_train / 2 + 0.5     # unnormalize
    img_train = img_train.numpy()
    lab_train = lab_train.numpy()
    
    dataiter = iter(testloader)
    img_test, lab_test = dataiter.next()
    img_test = img_test / 2 + 0.5     # unnormalize
    img_text = img_test.numpy()
    lab_test = lab_test.numpy()
    ac = []
    mc = []    
    
    x = np.arange(10,20,2)
    for clusters in x:
        sift = cv.SIFT_create()
        kmeans = KMeans(n_clusters=clusters, random_state=0)
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        
        des_train = []
        des_predi = []
    
        for i, img in enumerate(img_train):
            img =  (img* 255).astype(np.uint8)
            img = np.transpose(img, (1,2,0))
            kp, des = sift.detectAndCompute(img,None)
    
            if str(type(des)) == "<class 'NoneType'>":
                des = np.zeros((1,128))
            
            des = np.float64(des)
            des_predi.append(des)
            des_train.extend(des)
            
    
        kmeans.fit(des_train)
    
        dici_train = []
        for i, pred in enumerate(des_predi):
            pr = kmeans.predict(pred)
            f, _ = np.histogram(pr, bins = clusters, density = True)
            dici_train.append(f)
    
        clf.fit(dici_train, lab_train)
        
        des_test = []
        des_predi = []
        for j, img in enumerate(img_text):
            img =  (img* 255).astype(np.uint8)
            img = np.transpose(img, (1,2,0))
            kp, des = sift.detectAndCompute(img,None)
    
            if str(type(des)) == "<class 'NoneType'>":
                des = np.zeros((1,128))
           
            des = np.float64(des)
            des_predi.append(des)
            des_test.extend(des)
            
    
        dici_test = []
        for j, pred in enumerate(des_predi):
            pr = kmeans.predict(pred)
            f, _ = np.histogram(pr, bins = clusters, density = True)
            dici_test.append(f)
    
        y_pred = clf.predict(dici_test)
    
        ac.append(accuracy_score(lab_test, y_pred)) 
    
        mc.append(confusion_matrix(lab_test, y_pred))
        
        print('Cluster ' + str(clusters))
    
    plt.figure()
    plt.plot(x, ac)