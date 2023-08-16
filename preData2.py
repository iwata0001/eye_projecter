# メッシュ作成, データ成型

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

from mesh2Lib import mesh2
import preData as pre
import DmeshLib as DMesh
from utlLib import isExeption

# 分散共分散行列を求めるために、データを横に並べた行列を作る
# データの平均を求めてデータから引く（平均を0にする）


N = 143
newImgs = []

for i in range(N):
    if isExeption(i):
        newImgs.append("dummy")
        continue
    tex = cv2.imread('data_eyes/'+ str(i+1).zfill(3)+ '.png')
    eyeMesh = mesh2(64,48, tex)
    eyeMesh.setHandlesOrg(pre.handlesArr[i])
    eyeMesh.setHandlesDfm(pre.handlesAvg)
    eyeMesh.applyHandles()
    
    newImg = eyeMesh.deform()
    if i < 5:
        cv2.imwrite('temp_img/deform'+ str(i+1).zfill(3)+ '.png', newImg)
    newImgs.append(newImg)

#######################################################################################################################################

# 多重ウェーブレット版 データ加工してベクトル化
# 未使用
level = 2

datasArr = [[] for i in range(level+1)]

for i in range(N):
    
    if isExeption(i):
        for i in range(level+1):
            datasArr[i].append("")
        continue

    newImg = newImgs[i]
    
    newImg = cv2.cvtColor(newImg, cv2.COLOR_BGR2YCrCb)
    
    Y, Cr, Cb = cv2.split(newImg)
    
    datasY, shapesY = DMesh.freqDatas(Y, level)
    datasCr, shapesCr = DMesh.freqDatas(Cr, level)
    datasCb, shapesCb = DMesh.freqDatas(Cb, level)
    
    for i in range(level+1):
        data = np.concatenate([datasY[i], datasCr[i], datasCb[i]])
        datasArr[i].append(data)

LC = 1
MC = 1
HC = 1
handC = 100    #(LC+MC+HC) * 50
is_file = os.path.isfile('saves/eigValLev2_'+str(LC)+'_'+str(MC)+'_'+str(HC)+'_'+str(handC)+'.npy')

print(is_file)
dataArr = []
print(datasArr[0][0].shape, datasArr[1][0].shape, datasArr[2][0].shape, pre.handlesArr[0].reshape(pre.H*1*2).shape)
for i in range(N):
    if isExeption(i):
        continue
    data = np.concatenate([LC*datasArr[0][i], MC*datasArr[1][i], HC*datasArr[2][i], handC*pre.handlesArr[i].reshape(pre.H*1*2)])
    dataArr.append(data)
    
dataArr = np.array(dataArr)

avgData = np.sum(dataArr, axis=0) / (N-3)
print(len(avgData))
dataCentArr = []
for i in range(len(dataArr)):
    dataCentArr.append(dataArr[i] - avgData)
dataCentArr = np.array(dataCentArr)


######################################################################################################################################################################################

# 生画像データ加工 データ加工してベクトル化
# 画像＋特徴点
# 未使用
img1 = cv2.imread('data_eyes/001.png')
avgEye = np.zeros_like(img1)*(1.0)
eyeDatas = []
handleDatas = []
handCoeff = 150

for i in range(N):
    
    if isExeption(i):
        continue
    
    newImg = newImgs[i]
    
    avgEye += newImg/1
    
    eyeVec = newImg.reshape(48*64*3)
    
    handleVec = pre.handlesArr[i].reshape(pre.H*1*2) * handCoeff
    
    eyeData = np.append(eyeVec, handleVec)
    
    eyeDatas.append(eyeData)
    handleDatas.append(handleVec)
    
avgEye = avgEye / (N-3)
avgEye = avgEye.astype(np.uint8)
eyeDatas = np.array(eyeDatas)
avgEyeVec = avgEye.reshape(48*64*3)
handlesAvgVec = pre.handlesAvg.reshape(pre.H*1*2) * handCoeff
avgEyeData = np.append(avgEyeVec, handlesAvgVec)
eyeDatasCenter = eyeDatas - avgEyeData

#######################################################################################################################################################################################

# 生画像データ加工 データ加工してベクトル化
# 画像＋特徴点＋ベクタ画像（制御点）
img1 = cv2.imread('data_eyes/001.png')
avgEye_v1 = np.zeros_like(img1)*(1.0)
eyeDatas_v1 = []
handleDatas_v1 = []
vectorDatas_v1 = []
handCoeff_v1 = 150
vecCoeff = 10
eyeCoeff = 1

datanum = 0
for i in range(N):
    
    path = 'json_data/' + str(i+1).zfill(3) + '_v2.json'
    if not os.path.isfile(path):
        continue
    
    print(i+1)
    newImg_v1 = newImgs[i]
    
    avgEye_v1 += newImg_v1/1
    
    eyeVec = newImg_v1.reshape(48*64*3) * eyeCoeff
    
    handleVec = pre.handlesArr[i].reshape(pre.H*1*2) * handCoeff_v1

    vecdata = None
    with open('json_data/'+str(i+1).zfill(3)+'_v2.json') as f:
        vecdata = json.load(f)
    vectorVec = vecdata['shapeUOx']+ vecdata['shapeUOy']+ vecdata['shapeUIx']+ vecdata['shapeUIy']+ vecdata['shapeLOx']+ vecdata['shapeLOy']+ vecdata['shapeLIx']+ vecdata['shapeLIy']
    for j in range(len(vecdata['pplXY'])):
        vectorVec = vectorVec + vecdata['pplXY'][j]

    vectorVec = np.array(vectorVec)
    vectorVec = vectorVec * vecCoeff
    
    eyeData_v1 = np.append(eyeVec, handleVec)
    eyeData_v1 = np.append(eyeData_v1, vectorVec)
    
    eyeDatas_v1.append(eyeData_v1)
    handleDatas_v1.append(handleVec)

    datanum = datanum+1


eyeDatas_v1a = np.array(eyeDatas_v1)
#print(eyeDatas_v1a.shape)

avgdata_v1 = np.mean(eyeDatas_v1a, axis=0)
#print(avgdata_v1.shape)

eyeDatasCenter_v1a = eyeDatas_v1a - avgdata_v1
#print(eyeDatasCenter_v1a)