# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 09:59:24 2021

@author: leona
"""
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np



dataset = './set2/'
files = os.listdir(dataset)

cont = 0

a = 4
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
bf = cv.BFMatcher()
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
sift = cv.SIFT_create()

i = 0
src_file = files[i]

src_img = cv.imread(dataset + src_file)
src_img = cv.cvtColor(src_img,cv.COLOR_BGR2RGB)
wi,hi,di = src_img.shape
wi = int(wi/a)
hi = int(hi/a)
src_img = cv.resize(src_img,(int(hi), int(wi)))

files.remove(files[i])

# print('Plot')
# plt.figure(cont)
# cont = cont + 1
# plt.imshow(src_img)
# plt.show()

# while len(files) != 0:
for j, dst_file in enumerate(files):
    
    print('test ' + src_file + str(' whit ') + dst_file )
    

    dst_img = cv.imread(dataset + dst_file)
    dst_img = cv.cvtColor(dst_img,cv.COLOR_BGR2RGB)
    wj,hj,dj = dst_img.shape
    wj = int(wj/a)
    hj = int(hj/a)
    dst_img = cv.resize(dst_img,(int(hj), int(wj)))
    
    
    print('Begin Sifit', i, j)
    src_kp, src_des = sift.detectAndCompute(src_img,None)
    dst_kp, dst_des = sift.detectAndCompute(dst_img,None)
    
    print('Match')
    matches = bf.knnMatch(src_des,dst_des, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    print(len(good))
    
    
    if len(good)>10:
        
        src_pts = np.float32([ src_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ dst_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        print('Homography')
        # M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,3.0)
        M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC,3.0)
        print('Warp')
        
       
        # M = src_M@np.linalg.inv(dst_M)
        
        warp_dst = cv.warpPerspective(dst_img, M, (hj*2, wj))
        # warp_dst = cv.warpPerspective(dst_img, M, (hj, wj))
        # warp_dst[0:src_img.shape[0], 0:src_img.shape[1]] = src_img
        
        
        
        
        
        print('Plot')
        plt.figure(cont)
        plt.title(src_file + ' ' + dst_file)
        cont = cont + 1
        plt.imshow(warp_dst)
        plt.show()
       
        # plt.figure(cont)
        # plt.title(str(i) + ' ' + str(j))
        # cont = cont + 1
        # plt.imshow(warp_dst)
        
        
        
        print('End Sifit', i, j)
        
        files.remove(files[j])
            
            # src_kp, src_des = brief.compute(src_img, src_kp)
            # dst_kp, dst_des = brief.compute(dst_img, dst_kp)
            
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

