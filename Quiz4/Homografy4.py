# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 09:59:24 2021

@author: leona
"""
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def extractsift(img):
    print('Begin Sifit')
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(img,None)
    
    return (kp, des)

def extractbrief(img):
    print('Begin Brief')
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    kp, _ = extractsift(img)
    kp, des = brief.compute(img, kp)
    return (kp, des)


def matches(src_des, dst_des):
    print('Match')
    bf = cv.BFMatcher_create()
    matches = bf.knnMatch(src_des,dst_des, k=2)
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    print(len(good))
        
    return good


def homografy(good, src_kp, dst_kp):
    print('Homography')
    src_pts = np.float32([ src_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ dst_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC,3.0)
    return M, mask

def warp(dst_img, src_img, M, blend):
    
    print('Warp')
    dst_w, dst_h, _ = dst_img.shape
    src_w, src_h, _ = src_img.shape
    
    
    warp_dst = cv.warpPerspective(dst_img, M, (src_h*2, src_w*2))   
    
    if blend == 0:
        print('normal')
        warp_dst[0:src_img.shape[0], 0:src_img.shape[1]] = src_img
         
        
    warp_dst = np.float32(warp_dst)
    src_img = np.float32(src_img)
    warp_w, warp_h, _ = warp_dst.shape    
    
    if blend == 1:
        print('averaging')
        for x in np.arange(0,src_w):
            for y in np.arange(0, src_h):
                for d in np.arange(0,3):
                    if (warp_dst[x,y,d] + src_img[x,y,d] == src_img[x,y,d]):
                        warp_dst[x,y,d] = src_img[x,y,d]
                    if warp_dst[x,y,d] + src_img[x,y,d] > src_img[x,y,d]:
                        warp_dst[x,y,d] = (warp_dst[x,y,d] + src_img[x,y,d])/2
     
    if blend == 2:
        print('averaging')
        
        for y in np.arange(0, src_h):
            for d in np.arange(0,3):
                w2 = 0
                for x in np.arange(0,src_w):
                    if (warp_dst[x,y,d] + src_img[x,y,d] == src_img[x,y,d]):
                        warp_dst[x,y,d] = src_img[x,y,d]
                    if warp_dst[x,y,d] + src_img[x,y,d] > src_img[x,y,d]:
                        w1 = x
                        w2 = w2 + 1
                        warp_dst[x,y,d] = (warp_dst[x,y,d]*w2 + src_img[x,y,d]*w1)/(w1+w2)

    return np.uint8(warp_dst)
    
    
    

dataset = './set1/'
files = os.listdir(dataset)

cont = 0
a = 4
i = 0
src_file = files[i]

src_img = cv.imread(dataset + src_file)
src_img = cv.cvtColor(src_img,cv.COLOR_BGR2RGB)
src_w,src_h,_ = src_img.shape
src_w = int(src_w/a)
src_h = int(src_h/a)
src_img = cv.resize(src_img,(int(src_h), int(src_w)))

files.remove(files[i])

while len(files) != 0:
    for j, dst_file in enumerate(files):
        
        print('test ' + src_file + str(' whit ') + dst_file )
        

        dst_img = cv.imread(dataset + dst_file)
        dst_img = cv.cvtColor(dst_img,cv.COLOR_BGR2RGB)
        dst_w,dst_h,_ = dst_img.shape
        dst_w = int(dst_w/a)
        dst_h = int(dst_h/a)
        dst_img = cv.resize(dst_img,(int(dst_h), int(dst_w)))
        
        src_kp, src_des = extractsift(src_img)
        dst_kp, dst_des = extractsift(dst_img)
        
        good = []
        good = matches(src_des, dst_des)
        
        print(len(good))
        
        maxpoint = 0
        if len(good)>maxpoint:
            M, _ = homografy(good, src_kp, dst_kp)
            src_img = warp(dst_img, src_img, M, 2)
            files.remove(dst_file)
            
         
plt.figure() 
plt.imshow(src_img)
plt.show()           
        
           
            
        
            
            
      
            # matches = bf.knnMatch(src_des,dst_des, k=2)
           
            # good = []
            # for m,n in matches:
            #     if m.distance < 0.7*n.distance:
            #         good.append(m)
            # # if len(good)>10:
            # src_pts = np.float32([ src_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            # dst_pts = np.float32([ dst_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            # Mm, maskm = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            
            # warp_dst1 = cv.warpPerspective(dst_img, M, (hj, wj))

