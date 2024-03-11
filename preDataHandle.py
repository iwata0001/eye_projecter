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
for i in range(0, 109):
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

# ランダムに拡大したデータを追加
"""
for d in featVecsNormalized_EM:
    img, han = np.split(d,[48*64])
    for i in range(9):
        rand = np.random.rand() + 0.5
        randVec = np.append(img, han*rand)
        randVec = np.array([randVec])
        featVecsNormalized_EM = np.append(featVecsNormalized_EM, randVec, axis=0)
avg = np.mean(featVecsNormalized_EM, axis=0)
# 新しい平均
featVecsNormalized_EM = featVecsNormalized_EM - avg
avgI, avgH = np.split(avg, [48*64])
avgImageVec = avgImageVec + avgI
avgHandleVec = avgHandleVec + avgH
print(featVecsNormalized_EM.shape, randVec.shape)
"""

#計算しなおすときはTF = Trueに
TF = False
def calcHandleEig():
    covMat = np.cov(featVecsNormalized_EM.T)
    print("covmat",covMat.shape)
    dataEig = np.linalg.eig(covMat)
    np.save('saves/eigVal_autoHandleGen_EMPCA', dataEig[0].real)
    np.save('saves/eigVec_autoHandleGen_EMPCA', dataEig[1].real.T)
path = 'saves/eigVal_autoHandleGen_EMPCA.npy'
if not os.path.isfile(path):
    print("ハンドル固有値再計算")
    calcHandleEig()