# 入力(画像+特徴点ハンドル+ベクタ画像制御点)を投影する関数project_withVectorを定義

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from ipywidgets import FloatSlider
import copy
import math
import pywt
import os
import random
import json

from mesh2Lib import mesh2
import preData as pre
import preData2 as pre2
import DmeshLib as DMesh

# 主成分分析 固有ベクトルを読み込んで固有値順に
eyeCoeff = 1
handCoeff = 150
vecCoeff = 10
eigVal = np.load('saves/eyedata_eigval_'+str(eyeCoeff)+'-'+str(handCoeff)+'-'+str(vecCoeff)+'.npy')
eigVec = np.load('saves/eyedata_eigvec_'+str(eyeCoeff)+'-'+str(handCoeff)+'-'+str(vecCoeff)+'.npy')
indices = np.argsort(eigVal)[::-1]
eigValSum = np.sum(eigVal)
#print("eigvalsum", eigValSum)

name = '017'
variation = ''

handles = DMesh.detectP('data_eyes_p/'+ name +'_p.png')
handles2 = DMesh.detectP('data_eyes_p2/'+ name +'_p2.png')
handles3 = DMesh.detectP('data_eyes_p3/'+ name +'_p3.png')

handles = np.append(handles, handles2, axis=0)
handles = np.append(handles, handles3, axis=0)
handles = np.append(handles, np.array([[[32.5, 24.5]]]), axis=0)

#print(handles.shape)

#累積寄与率がcontRateになるまで固有ベクトルを並べる
contRate = 0.8
A = []
temp = 0
for n in range(len(eigVal)):
    temp += eigVal[indices[n]]
    if temp / eigValSum > contRate:
        D = n
        break
print("model dimension", D)

for n in range(D):
    A.append(eigVec[indices[n]])

# 並べた固有ベクトルで張られる空間に点を正射影する行列Pを求める
A = np.array(A)
A = A.T
P = np.linalg.inv(A.T @ A) @ A.T


def project_withVector(texture, handles):
    #texture = cv2.imread('data_eyes/001.png')
    tex = texture
    eyeMesh = mesh2(64,48, tex)
    eyeMesh.setHandlesOrg(handles)
    eyeMesh.setHandlesDfm(pre.handlesAvg)
    eyeMesh.applyHandles()

    img = eyeMesh.deform()
    avgEyeData = pre2.avgdata_v1

    avgImgData, avgHandleData, avgVectorData = np.split(avgEyeData, [48*64*3, 48*64*3+13*1*2])
    #print(avgEyeData.shape, avgImgData.shape, avgHandleData.shape, avgVectorData.shape)
    vectorTemp = avgVectorData

    # 初期値を変更する
    for ind, elem in enumerate(vectorTemp):
        vectorTemp[ind] = vectorTemp[ind] #+ random.random()*1000-500
        #print(vectorTemp[ind])

    #初期値ベクターデータを保存
    N = 5
    vectorTemp = np.array(vectorTemp)
    vectorTempN = vectorTemp/vecCoeff
    UOx, UOy, UIx, UIy, LOx, LOy, LIx, LIy, ppl = np.split(vectorTempN, [N, 2*N, 3*N, 4*N, 5*N, 6*N, 7*N, 8*N])
    pplXY = []
    #print(len(ppl)/2)
    for j in range(int(len(ppl)/2)):
        pplXY.append([ppl[2*j], ppl[2*j+1]])

    vecdata = {}
    vecdata['shapeUIx'] = UIx.tolist()
    vecdata['shapeUIy'] = UIy.tolist()
    vecdata['shapeUOx'] = UOx.tolist()
    vecdata['shapeUOy'] = UOy.tolist()
    vecdata['shapeLIx'] = LIx.tolist()
    vecdata['shapeLIy'] = LIy.tolist()
    vecdata['shapeLOx'] = LOx.tolist()
    vecdata['shapeLOy'] = LOy.tolist()
    vecdata['pplXY'] = pplXY

    with open('json_data/fitting/vector_fitted_itr=0.json', 'w') as f:
        json.dump(vecdata, f)

    xs = []
    diffs = []
    for i in range(100):
        eyeVec = img.reshape(48*64*3) * eyeCoeff
        handleVec = handles.reshape(pre.H*1*2) * handCoeff

        # 画像(eyeVec)、特徴点(handleVec)、ベクターデータ(vectorTemp)をくっつける
        eyeData = np.append(eyeVec, handleVec)
        eyeData = np.append(eyeData, vectorTemp)

        # 平均が０になるように正規化
        eyeDataCenter = eyeData - avgEyeData

        # 投影
        if D != 0:
            x = P @ eyeDataCenter
            p = A @ x

            p = avgEyeData + p
            newImg, newHandles, newVector = np.split(p, [48*64*3, 48*64*3+13*1*2])
        else:
            p = avgEyeData
            newImg, newHandles, newVector = np.split(p, [48*64*3, 48*64*3+13*1*2])

        #入力と出力の二乗誤差(特徴点)
        ioDiff = np.mean((handleVec-newHandles)**2 / (handCoeff**2))
        #normDiff = np.linalg.norm(np.array(vectorTemp) - np.array(newVector))
        print("ioDiff:",ioDiff)
        diffs.append(ioDiff)
        xs.append(i)

        newImg = newImg.reshape(48,64,3)
        newImg = np.clip(newImg, 0, 255)
        newImg = newImg.astype(np.uint8)

        newHandlesTemp = newHandles / handCoeff
        newHandlesTemp = newHandlesTemp.reshape(pre.H,1,2)

        newMesh = mesh2(64,48, newImg)
        newMesh.setHandlesOrg(pre.handlesAvg)
        newMesh.setHandlesDfm(newHandlesTemp)
        #newMesh.setHandlesDfm(handles)
        newMesh.applyHandles()

        newEye = newMesh.deform()
        cv2.imwrite('output/fitting_img/'+'img_itr'+str(i+1)+'.png', newEye)


        vectorTemp = newVector

        #ベクターデータを保存
        N = 5
        newVectorN = newVector/vecCoeff
        UOx, UOy, UIx, UIy, LOx, LOy, LIx, LIy, ppl = np.split(newVectorN, [N, 2*N, 3*N, 4*N, 5*N, 6*N, 7*N, 8*N])
        pplXY = []
        #print(len(ppl)/2)
        for j in range(int(len(ppl)/2)):
            pplXY.append([ppl[2*j], ppl[2*j+1]])

        vecdata = {}
        vecdata['shapeUIx'] = UIx.tolist()
        vecdata['shapeUIy'] = UIy.tolist()
        vecdata['shapeUOx'] = UOx.tolist()
        vecdata['shapeUOy'] = UOy.tolist()
        vecdata['shapeLIx'] = LIx.tolist()
        vecdata['shapeLIy'] = LIy.tolist()
        vecdata['shapeLOx'] = LOx.tolist()
        vecdata['shapeLOy'] = LOy.tolist()
        vecdata['pplXY'] = pplXY

        with open('json_data/fitting/vector_fitted_itr='+str(i+1)+'.json', 'w') as f:
            json.dump(vecdata, f)
        

    plt.plot(xs, diffs)
    plt.show()
    
    newImg = newImg / eyeCoeff
    newImg = newImg.reshape(48,64,3)
    newImg = np.clip(newImg, 0, 255)
    newImg = newImg.astype(np.uint8)

    newHandles = newHandles / handCoeff
    newHandles = newHandles.reshape(pre.H,1,2)

    newMesh = mesh2(64,48, newImg)
    newMesh.setHandlesOrg(pre.handlesAvg)
    newMesh.setHandlesDfm(newHandles)
    #newMesh.setHandlesDfm(handles)
    newMesh.applyHandles()

    newEye = newMesh.deform()

    #cv2.imwrite('output/'+ name + variation + '_projected_'+str(contRate)+'.png', newEye)

    N = 5
    newVectorN = newVector/vecCoeff
    UOx, UOy, UIx, UIy, LOx, LOy, LIx, LIy, ppl = np.split(newVectorN, [N, 2*N, 3*N, 4*N, 5*N, 6*N, 7*N, 8*N])
    pplXY = []
    #print(len(ppl)/2)
    for i in range(int(len(ppl)/2)):
        pplXY.append([ppl[2*i], ppl[2*i+1]])


    vecdata = {}
    vecdata['shapeUIx'] = UIx.tolist()
    vecdata['shapeUIy'] = UIy.tolist()
    vecdata['shapeUOx'] = UOx.tolist()
    vecdata['shapeUOy'] = UOy.tolist()
    vecdata['shapeLIx'] = LIx.tolist()
    vecdata['shapeLIy'] = LIy.tolist()
    vecdata['shapeLOx'] = LOx.tolist()
    vecdata['shapeLOy'] = LOy.tolist()
    vecdata['pplXY'] = pplXY

    with open('json_data/avg_v2.json', 'w') as f:
        json.dump(vecdata, f)
    print("saved.")

    plt.imshow(cv2.cvtColor(newEye, cv2.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
    plt.show()

    return newEye, newHandles