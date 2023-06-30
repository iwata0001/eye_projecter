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
imageVecs = []
handleVecs = []
for i in range(N):
    if isExeption(i):
        #newImgs.append(None)
        #newHandles.append(None)
        continue

    handle = preData.handlesArr[i]
    handleVec = handle.reshape(preData.H*1*2)
    handleVecs.append(handleVec)

    img = cv2.imread('data_eyes/'+ str(i+1).zfill(3)+ '.png')
    #ここで画像を変換（白黒？微分？）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgVec = img.reshape(48*64)
    imageVecs.append(imgVec)

imageVecs = np.array(imageVecs)
avgImageVec = np.mean(imageVecs, axis=0)
imageVecsNormalized = imageVecs - avgImageVec

handleVecs = np.array(handleVecs)
avgHandleVec = np.mean(handleVecs, axis=0)
handleVecsNormalized = handleVecs - avgHandleVec

#計算しなおすときはコメントを外す
covMat = np.cov(imageVecsNormalized.T)
print("covmat",covMat.shape)
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

imageA = imageVecsNormalized.T
imageP = np.linalg.inv(imageA.T @ imageA) @ imageA.T

def project_autoHandleGen(sketch): #skechはcv2.imreadで読み込んだもの
    imgVec = sketch.reshape(48*64)
    imgVecNormalized = imgVec - avgImageVec

    if D != 0:
        x = P @ imgVecNormalized
        p = A @ x

        newImgVec = avgImageVec + p
    else:
        newImgVec = avgImageVec

    image_x = imageP @ p
    newHandleVec = (handleVecsNormalized.T) @ image_x + avgHandleVec
    #newHandleVec.reshape(preData.H,1,2)

    print(newHandleVec)

    return(newImgVec, newHandleVec)

sketch = cv2.imread('temp_img/out.png')
sketch = cv2.resize(sketch, dsize=(64, 48))
sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
newImgVec= project_autoHandleGen(sketch)

#newImg = newImgVec.reshape(48,64)
#newImg = np.clip(newImg, 0, 255)
#newImg = newImg.astype(np.uint8)
#cv2.imshow("image", newImg)
#cv2.waitKey()

