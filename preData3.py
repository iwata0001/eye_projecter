import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from ipywidgets import FloatSlider
import copy
import math
import pywt
import os
import json

from utlLib import isExeption
from mesh2Lib import mesh2
import preData

N = 143
coeff = 10
newImgs = []
newHandles = []
featureVecs = []
for i in range(N):
    if isExeption(i):
        #newImgs.append(None)
        #newHandles.append(None)
        continue
    img = cv2.imread('data_eyes/'+ str(i+1).zfill(3)+ '.png')
    handle = preData.handlesArr[i]

    #ここで画像を変換（白黒？微分？）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgVec = img.reshape(48*64)
    handleVec = handle.reshape(preData.H*1*2)

    featureVec = np.append(imgVec, handleVec*coeff)
    featureVecs.append(featureVec)

featureVecs = np.array(featureVecs)
avgFeatureVec = np.mean(featureVecs, axis=0)
print(avgFeatureVec)
featureVecsNormalized = featureVecs - avgFeatureVec

avgImgVec, avgHandleVec = np.split(avgFeatureVec, [48*64])
print(avgImgVec.shape, avgHandleVec.shape)

#計算しなおすときはコメントを外す
#covMat = np.cov(featureVecsNormalized.T)
#dataEig = np.linalg.eig(covMat)
#np.save('saves/eigVal_autoHandleGen', dataEig[0].real)
#np.save('saves/eigVec_autoHandleGen', dataEig[1].real.T)

eigVal = np.load('saves/eigVal_autoHandleGen.npy')
eigVec = np.load('saves/eigVec_autoHandleGen.npy')
indices = np.argsort(eigVal)[::-1]
eigValSum = np.sum(eigVal)

#累積寄与率がcontRateになるまで固有ベクトルを並べる
contRate = 0.8
A = []
temp = 0
for n in range(len(eigVal)):
    temp += eigVal[indices[n]]
    if temp / eigValSum > contRate:
        D = n
        break

print("D: ", D)

for n in range(D):
    A.append(eigVec[indices[n]])

# 並べた固有ベクトルで張られる空間に点を正射影する行列Pを求める
A = np.array(A)
A = A.T
P = np.linalg.inv(A.T @ A) @ A.T

def project_autoHandleGen(sketch): #skechはcv2.imreadで読み込んだもの
    imgVec = sketch.reshape(48*64)

    featureVec = np.append(imgVec, avgHandleVec)

    featureVecNormalized = featureVec - avgFeatureVec

    if D != 0:
        tempFeatureVec = featureVecNormalized
        newHandleVec = np.array([0])

        for i in range(100):
            if all(newHandleVec) != 0:
                tempFeatureVec = np.append(imgVec, newHandleVec)
                tempFeatureVec = tempFeatureVec - avgFeatureVec
                print(i)
            x = P @ tempFeatureVec
            p = A @ x

            p = avgFeatureVec + p
            newImgVec, newHandleVec = np.split(p, [48*64])
    else:
        p = avgFeatureVec
        newImgVec, newHandleVec = np.split(p, [48*64])

    return(newImgVec, newHandleVec/coeff)

#sketch = cv2.imread('temp_img/out.png')
#sketch = cv2.resize(sketch, dsize=(64, 48))
#sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
#newImgVec, newHandleVec = project_autoHandleGen(sketch)

#newImg = newImgVec.reshape(48,64)
#newImg = np.clip(newImg, 0, 255)
#newImg = newImg.astype(np.uint8)
#cv2.imshow("image", newImg)
#cv2.waitKey()

