# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 14:10:18 2021

@author: leona
"""

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense



from tensorflow.keras import datasets, layers, models


import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import numpy as np
import cv2 as cv


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


l = 1000
train_images = train_images[0:l]
train_labels = train_labels[0:l]

l = 100
test_images = test_images[0:l]
test_labels = test_labels[0:l]

def bof(img_train, img_test, clusters):

    dict_train = []
    dict_test = []
    for cluster in clusters:
        f_train = []
        f_test = []
        sift = cv.SIFT_create()
        kmeans = KMeans(n_clusters=cluster, random_state=0)

        des_train = []
        des_train_pred = []
        des_test = []
        des_test_pred = []

        for img_tr in img_train:
            kp, des = sift.detectAndCompute(img_tr,None)

            if str(type(des)) == "<class 'NoneType'>":
                des = np.zeros((1,128))

            des = np.float64(des)
            des_train_pred.append(des)
            des_train.extend(des)

        kmeans.fit(des_train)

        for i, pred in enumerate(des_train_pred):
            pr = kmeans.predict(pred)
            freq, _ = np.histogram(pr, bins = cluster, density = True)
            f_train.append(freq)

        dict_train.append(f_train)

        for img_te in img_test:
            kp, des = sift.detectAndCompute(img_te,None)

            if str(type(des)) == "<class 'NoneType'>":
                des = np.zeros((1,128))

            des = np.float64(des)
            des_test_pred.append(des)
            des_test.extend(des)


        for j, pred in enumerate(des_test_pred):
            pr = kmeans.predict(pred)
            freq, _ = np.histogram(pr, bins = cluster, density = True)
            f_test.append(freq)

        dict_test.append(f_test)

        print('Cluster ' + str(cluster))

    return(dict_train, dict_test)


clusters = np.arange(10,12,2)
dict_train, dict_test = bof(train_images, test_images, clusters)

model = models.Sequential()

# model.add(layers.Flatten())
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(10, activation='relu'))

model.summary

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(np.array(dict_train[0]), train_labels, epochs=200, 
                    validation_data=(np.array(dict_test[0]), test_labels))




# # Normalize pixel values to be between 0 and 1
# train_images, test_images = train_images / 255.0, test_images / 255.0

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10))

# model.summary()


# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# history = model.fit(train_images, train_labels, epochs=10, 
#                     validation_data=(test_images, test_labels))

def cnn(train_images, train_labels, test_images, test_labels):

    # (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

    model = models.Sequential()
    model.add(layers.Input(5,1,1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.summary()


    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', input_shape=(5,)))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(dict_traint, lab_traint, epochs=10,
                        validation_data=(dict_testt, lab_test))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)