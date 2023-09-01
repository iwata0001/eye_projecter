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
featureVecs = []
avgHandle = preData.handlesAvg
for i in range(N):
    if isExeption(i):
        #newImgs.append(None)
        #newHandles.append(None)
        continue

    handle = preData.handlesArr[i]
    handleVec = handle.reshape(preData.H*1*2)
    handleVecs.append(handleVec)

    img = cv2.imread('data_eyes/'+ str(i+1).zfill(3)+ '.png')

    #ここで画像を変換（白黒, トリミング, 変形）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mesh = mesh2(64,48,img)
    hndl = np.array([handle[1],handle[2],handle[3],handle[12]])
    avgHndl = np.array([avgHandle[1],avgHandle[2],avgHandle[3],avgHandle[12]])
    mesh.setHandlesOrg(hndl, avgHndl)
    mesh.setHandlesDfm(avgHndl)
    mesh.applyHandles()
    imgNormalized = mesh.deform()

    if i<5:
        cv2.imwrite("temp_img/deformImg"+str(i+1)+".png", imgNormalized)

    imgVec = imgNormalized.reshape(48*64)
    imageVecs.append(imgVec)

imageVecs = np.array(imageVecs)
avgImageVec = np.mean(imageVecs, axis=0)
imageVecsNormalized = imageVecs - avgImageVec
print(np.std(imageVecsNormalized))

handleVecs = np.array(handleVecs)
avgHandleVec = np.mean(handleVecs, axis=0)
handleVecsNormalized = handleVecs - avgHandleVec

handleVecs_4 = np.delete(handleVecsNormalized, [0,1,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], 1)
avgHandleVec_4 = avgHndl.reshape(4*1*2)

# ハンドル4つでそのままのほうのデータ
stdImage = np.std(imageVecsNormalized)
stdHandle = np.std(handleVecs_4) /10
featVecsNormalized = np.append(imageVecsNormalized/stdImage, handleVecs_4/stdHandle, axis=1)

# EM-PCAをやる用のデータ
stdHandle_EM = np.std(handleVecsNormalized) /10
featVecsNormalized_EM = np.append(imageVecsNormalized/stdImage, handleVecsNormalized/stdHandle_EM, axis=1)

#計算しなおすときはTrueに
if 0:
    covMat = np.cov(featVecsNormalized.T)
    print("covmat",covMat.shape)
    dataEig = np.linalg.eig(covMat)
    np.save('saves/eigVal_autoHandleGen_4handle', dataEig[0].real)
    np.save('saves/eigVec_autoHandleGen_4handle', dataEig[1].real.T)

if 0:
    covMat = np.cov(featVecsNormalized_EM.T)
    print("covmat",covMat.shape)
    dataEig = np.linalg.eig(covMat)
    np.save('saves/eigVal_autoHandleGen_EMPCA', dataEig[0].real)
    np.save('saves/eigVec_autoHandleGen_EMPCA', dataEig[1].real.T)

eigVal = np.load('saves/eigVal_autoHandleGen_4handle.npy')
eigVec = np.load('saves/eigVec_autoHandleGen_4handle.npy')
indices = np.argsort(eigVal)[::-1]
eigValSum = np.sum(eigVal)

eigVal_EM = np.load('saves/eigVal_autoHandleGen_EMPCA.npy')
eigVec_EM = np.load('saves/eigVec_autoHandleGen_EMPCA.npy')
indices_EM = np.argsort(eigVal_EM)[::-1]
eigValSum_EM = np.sum(eigVal_EM)

#累積寄与率がcontRateになるまで固有ベクトルを並べる
contRate = 0.8
A = []
temp = 0
for n in range(len(eigVal)):
    temp += eigVal[indices[n]]
    if temp / eigValSum > contRate:
        D = n
        break

#print("D: ", D)

for n in range(D):
    A.append(eigVec[indices[n]])

A_EM = []
temp_EM = 0
for n in range(len(eigVal_EM)):
    temp_EM += eigVal_EM[indices_EM[n]]
    if temp_EM / eigValSum_EM > contRate:
        D_EM = n
        break

for n in range(D_EM):
    A_EM.append(eigVec_EM[indices_EM[n]])

# 並べた固有ベクトルで張られる空間に点を正射影する行列Pを求める
A = np.array(A)
A = A.T
P = np.linalg.inv(A.T @ A) @ A.T

A_EM = np.array(A_EM)
A_EM = A_EM.T
P_EM = np.linalg.inv(A_EM.T @ A_EM) @ A_EM.T

featA = featVecsNormalized.T
featP = np.linalg.inv(featA.T @ featA) @ featA.T

def findEdge(img):
    shape = img.shape
    black = np.zeros_like(img)
    edgeR = [0,0]
    edgeNumR = 0
    edgeL = [0,0]
    edgeNumL = 0
    edgeD = [0,0]
    edgeNumD = 0
    for j in range(shape[1]):
        for i in range(shape[0]):
            if img[i][j] != 255 and edgeNumL == 0:
                edgeNumL +=1
                edgeL[0] +=i
                edgeL[1] +=j
            if img[i][shape[1]-1-j] != 255 and edgeNumR == 0:
                black[i][shape[1]-1-j] = 255
                edgeNumR +=1
                edgeR[0] +=i
                edgeR[1] += (shape[1]-1-j)
        if (edgeNumR > 0) and (edgeNumL > 0):
            break

        for i in range(shape[0]):
            for j in range(shape[1]):
                if img[shape[0]-1-i][j] != 255:
                    edgeNumD +=1
                    edgeD[0] += (shape[0]-1-i)
                    edgeD[1] +=j
            if edgeNumD > 0:
                break

    return {"R":np.array([edgeR[1]/edgeNumR, edgeR[0]/edgeNumR]), 
            "L":np.array([edgeL[1]/edgeNumL, edgeL[0]/edgeNumL]),
            "D":np.array([edgeD[1]/edgeNumD, edgeD[0]/edgeNumD])}

def project_autoHandleGen(sketch,handle=avgHndl): #skechはcv2.imreadで読み込んだもの
    mesh = mesh2(64,48,sketch)
    mesh.setHandlesOrg(handle, avgHndl)
    mesh.setHandlesDfm(avgHndl)
    mesh.applyHandles()
    img = mesh.deform()
    imgVec = img.reshape(48*64)
    imgVecNormalized = imgVec - avgImageVec

    handleVec = handle.reshape(4*1*2)
    handleVecNormalized = handleVec - avgHandleVec_4

    featureVec = np.append(imgVecNormalized/stdImage, handleVecNormalized/stdHandle)
    avgFeatureVec = np.append(avgImageVec/stdImage, avgHandleVec_4/stdHandle)

    if D != 0:
        x = P @ featureVec
        p = A @ x

        newFeatVec = avgFeatureVec + p
    else:
        newFeatVec = avgFeatureVec

    newImgVec,nokori = np.split(newFeatVec,[48*64])
    feat_x = featP @ p
    newHandleVec = (handleVecsNormalized.T) @ feat_x + avgHandleVec
    #newHandleVec.reshape(preData.H,1,2)

    return(newImgVec*stdImage, newHandleVec)

def project_autoHandleGen_EM(sketch,handle=avgHndl): #skechはcv2.imreadで読み込んだもの
    mesh = mesh2(64,48,sketch)
    mesh.setHandlesOrg(handle, avgHndl)
    mesh.setHandlesDfm(avgHndl)
    mesh.applyHandles()
    img = mesh.deform()
    cv2.imwrite('temp_img/normalizedSketch_gray.png', img)
    imgVec = img.reshape(48*64)
    imgVecNormalized = imgVec - avgImageVec

    handleVec = handle.reshape(4*1*2)
    handleVecNormalized = handleVec - avgHandleVec_4
    initInputHandle = handleVecNormalized/stdHandle_EM

    initInputImg = imgVecNormalized/stdImage
    avgFeatureVec = np.append(avgImageVec/stdImage, avgHandleVec/stdHandle_EM)

    estimated = np.zeros(13*1*2)

    for i in range(100):
        estimated[[2,3,4,5,6,7,24,25]] = initInputHandle
        tempFeatureVec = np.append(initInputImg, estimated) # EMPCAのために最初は平均（０）で残りの次元を埋める　あとは推定値で埋める
        if D_EM != 0:
            x = P_EM @ tempFeatureVec
            p = A_EM @ x

            newFeatVec = avgFeatureVec + p
        else:
            newFeatVec = avgFeatureVec

        nokori, estimated = np.split(p,[48*64])

    newImgVec, newHandleVec = np.split(newFeatVec,[48*64])

    return(newImgVec*stdImage, newHandleVec*stdHandle_EM)

sketch = cv2.imread('temp_img/out.png')
sketch = cv2.resize(sketch, dsize=(64, 48))
sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
edge = findEdge(sketch)
cv2.drawMarker(sketch, (int(edge["R"][0]), int(edge["R"][1])), (0,0,0))
cv2.drawMarker(sketch, (int(edge["L"][0]), int(edge["L"][1])), (0,0,0))
cv2.drawMarker(sketch, (int(edge["D"][0]), int(edge["D"][1])), (0,0,0))

edgeR = np.array(edge["R"])
edgeD = np.array(edge["D"])
edgeL = np.array(edge["L"])

edgeR = np.array([edgeR])
edgeD = np.array([edgeD])
edgeL = np.array([edgeL])

handles = np.array([edgeR,edgeD,edgeL,np.array([[32.5,24.5]])])

newImgVec,newHandleVec= project_autoHandleGen(sketch, handles)
print(newHandleVec)
newImgVec,newHandleVec= project_autoHandleGen_EM(sketch, handles)
print(newHandleVec)

#newImg = newImgVec.reshape(48,64)
#newImg = np.clip(newImg, 0, 255)
#newImg = newImg.astype(np.uint8)
#cv2.imshow("image", sketch)
#cv2.waitKey()

