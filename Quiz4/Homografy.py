# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 09:59:24 2021

@author: leona
"""
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

files = os.listdir('./set1/')
key = {'brief': [], 'sift': []}
desc = {'brief': [], 'sift': []}

# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
bf = cv.BFMatcher()
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
sift = cv.SIFT_create()

for i, file in enumerate(files):
    
    img = cv.imread('./set1/'+ str(file))
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    w,h,d = img.shape
    img = cv.resize(img,(int(h/2), int(w/2)))
    # plt.figure(i)
    # plt.imshow(img)
    # plt.show()
    
    kp, des = sift.detectAndCompute(img,None)
    key['sift'].append(kp)
    desc['sift'].append(des)

    kp, des = brief.compute(img, kp)
    key['brief'].append(kp)
    desc['brief'].append(des)



# for m in ['brief', 'sift']:  
    
    
mask = {'brief': [], 'sift': []}
M = {'brief': [], 'sift': []}

for me in ['brief', 'sift']:
    for i, desci in enumerate(desc[me]):
        M[me].append([])
        for j, descj in enumerate(desc[me]):
            if i<j:
                print('begin', me,i,j)
                # matches = bf.match(desci,descj)
                matches = bf.knnMatch(desci,descj, k=2)
           
                good = []
                for m,n in matches:
                    if m.distance < 0.7*n.distance:
                        good.append(m)
                # if len(good)>10:
                src_pts = np.float32([ key[me][i][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ key[me][j][m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                Mm, maskm = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
                
                warp_dst1 = cv.warpPerspective(img2, M[me][i][j], (h2, w2))
                warp_dst2 = cv.warpPerspective(img2, M[me][i][j], (h1, w1))
                
                # M[me][i].append(Mm)
                # mask[me].append(maskm)
                print('end', me,i,j)
                
                
                # else:
                #     print( "Not enough matches are found - {}/{}".format(len(good),10) )
                #     matchesMask = None
                
                




cont = 0
mm = 0 
for i, file1 in enumerate(files):
    for j, file2 in enumerate(files):
        
            
        img1 = cv.imread('./set1/'+ str(file1))
        img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB)
        img1 = cv.resize(img1,(int(h/2), int(w/2)))
        img2 = cv.imread('./set1/'+ str(file2))
        img2 = cv.cvtColor(img2,cv.COLOR_BGR2RGB)
        img2 = cv.resize(img2,(int(h/2), int(w/2)))
        
        w1,h1,d1 = img1.shape
        w2,h2,d2 = img2.shape
        
        
        for me in ['brief']:
           print(i,j, me, file1, file2)
           warp_dst1 = cv.warpPerspective(img1, M[me][i][j], (h1, w1))
           warp_dst2 = cv.warpPerspective(img2, M[me][i][j], (h1, w1))
           # warp_dst1[0:img1.shape[0], 0:img1.shape[1]] = img1
           plt.figure(cont)
           plt.title(str(i) + ' ' + str(j))
           cont = cont + 1
           plt.imshow(warp_dst1)
           
           plt.figure(cont)
           plt.title(str(i) + ' ' + str(j))
           cont = cont + 1
           plt.imshow(warp_dst2)
              
               
               
               
                






# warp_mat = cv.getAffineTransform(srcTri, dstTri)
# warp_dst = cv.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))












# for m in ['brief', 'sift']:
#     for i, kpi in enumerate(keyp[m]):
#         for j, kpj in enumerate(keyp[m]):
#             if i!=j:
#                 for k, t enumerate()
#                     print(m,i,j)
#                     M, mask = cv.findHomography(kpi.pt, kpj.pt, cv.RANSAC, 3)

# for i, kpi in enumerate(keyp['brief']):
#     for j, kpj in enumerate(keyp['brief']):
#         if i!=j:
#             matches = bf.match(desci,descj)
#             M, mask = cv.findHomography(kpi.pt, kpj.pt, cv.RANSAC, 3)
   
    
   
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('simple.jpg',0)
# # Initiate FAST detector
# star = cv.xfeatures2d.StarDetector_create()
# # Initiate BRIEF extractor
# brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
# # find the keypoints with STAR
# kp = star.detect(img,None)
# # compute the descriptors with BRIEF
# kp, des = brief.compute(img, kp)
# print( brief.descriptorSize() )
# print( des.shape())    
    