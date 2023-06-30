# 入力データ(画像＋特徴点ハンドル)を投影する関数projectの定義
# 投影してリファレンスの詳細を移植するproject_addDetailの定義

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

from mesh2Lib import mesh2
import preData as pre
import preData2 as pre2
import DmeshLib as DMesh

# 投影

# 主成分分析 固有ベクトルを読み込んで固有値順に
eigVal = np.load('saves/eye_eig_val_13p_mesh_coef'+str(pre2.handCoeff)+'.npy')
eigVec = np.load('saves/eye_eig_vec_13p_mesh_coef'+str(pre2.handCoeff)+'.npy')
indices = np.argsort(eigVal)[::-1]
eigValSum = np.sum(eigVal)

name = '017'
variation = ''

handles = DMesh.detectP('data_eyes_p/'+ name +'_p.png')
handles2 = DMesh.detectP('data_eyes_p2/'+ name +'_p2.png')
handles3 = DMesh.detectP('data_eyes_p3/'+ name +'_p3.png')

handles = np.append(handles, handles2, axis=0)
handles = np.append(handles, handles3, axis=0)
handles = np.append(handles, np.array([[[32.5, 24.5]]]), axis=0)

print(handles.shape)

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

'''
name = ['002', 'rough_eye1']
variation = ['', '']
orgImg = [0,0]
prjImg = [0,0]
hands = [0,0]


for i in range(2):
    tex = cv2.imread('data_eyes/'+ name[i] + variation[i] + '.png')

    handles = DMesh.detectP('data_eyes_p/'+ name[i] +'_p.png')
    handles2 = DMesh.detectP('data_eyes_p2/'+ name[i] +'_p2.png')
    handles3 = DMesh.detectP('data_eyes_p3/'+ name[i] +'_p3.png')

    handles = np.append(handles, handles2, axis=0)
    handles = np.append(handles, handles3, axis=0)
    handles = np.append(handles, np.array([[[32.5, 24.5]]]), axis=0)

    eyeMesh = mesh2(64,48, tex)
    eyeMesh.setHandlesOrg(handles)
    eyeMesh.setHandlesDfm(pre.handlesAvg)
    eyeMesh.applyHandles()

    newImg = eyeMesh.deform()

    orgImg[i] = newImg

    #plt.imshow(cv2.cvtColor(newImg, cv2.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
    #plt.show()

    avgEyeData = pre2.avgEyeData

    eyeVec = newImg.reshape(48*64*3)
    handleVec = handles.reshape(pre.H*1*2) * pre2.handCoeff

    eyeData = np.append(eyeVec, handleVec)

    eyeDataCenter = eyeData - avgEyeData

    if D != 0:
        x = P @ eyeDataCenter
        p = A @ x

        p = avgEyeData + p
        newImg, newHandles = np.split(p, [48*64*3])
    else:
        p = avgEyeData
        newImg, newHandles = np.split(p, [48*64*3])



    newImg = newImg.reshape(48,64,3)
    newImg = np.clip(newImg, 0, 255)
    newImg = newImg.astype(np.uint8)

    prjImg[i] = newImg

    newHandles = newHandles / pre2.handCoeff
    newHandles = newHandles.reshape(pre.H,1,2)
    hands[i] = newHandles

    newMesh = mesh2(64,48, newImg)
    newMesh.setHandlesOrg(pre.handlesAvg)
    newMesh.setHandlesDfm(newHandles)
    #newMesh.setHandlesDfm(handles)
    newMesh.applyHandles()

    newEye = newMesh.deform()

diff = orgImg[0]/1 - prjImg[0]/1
newImg = diff/1 + prjImg[1]/1


diffD = np.clip(diff+128, 0, 255)
diffD = diffD.astype(np.uint8)
plt.imshow(cv2.cvtColor(prjImg[1], cv2.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
plt.show()
cv2.imwrite('output/diffProj/'+ name[1] + '_prj.png', prjImg[1])

newImg = np.clip(newImg, 0, 255)
newImg = newImg.astype(np.uint8)

newMesh = mesh2(64,48, newImg)
newMesh.setHandlesOrg(pre.handlesAvg)
newMesh.setHandlesDfm(hands[1])
#newMesh.setHandlesDfm(handles)
newMesh.applyHandles()

newEye = newMesh.deform()

#cv2.imwrite('output/'+ name + variation + '_projected_'+str(contRate)+'.png', newEye)

cv2.imwrite('output/diffProj/'+ name[1] + '_prj+'+ name[0] +'.png', newEye)
plt.imshow(cv2.cvtColor(newEye, cv2.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
plt.show()
'''


def project(texture, handles): # 目の画像とハンドルをベクトル化し, 空間に投影
    tex = texture
    eyeMesh = mesh2(64,48, tex)
    eyeMesh.setHandlesOrg(handles)
    eyeMesh.setHandlesDfm(pre.handlesAvg)
    eyeMesh.applyHandles()

    newImg = eyeMesh.deform()
    avgEyeData = pre2.avgEyeData

    eyeVec = newImg.reshape(48*64*3)
    handleVec = handles.reshape(pre.H*1*2) * pre2.handCoeff

    eyeData = np.append(eyeVec, handleVec)

    eyeDataCenter = eyeData - avgEyeData



    if D != 0:
        x = P @ eyeDataCenter
        p = A @ x

        p = avgEyeData + p
        newImg, newHandles = np.split(p, [48*64*3])
    else:
        p = avgEyeData
        newImg, newHandles = np.split(p, [48*64*3])



    newImg = newImg.reshape(48,64,3)
    newImg = np.clip(newImg, 0, 255)
    newImg = newImg.astype(np.uint8)

    newHandles = newHandles / pre2.handCoeff
    newHandles = newHandles.reshape(pre.H,1,2)

    newMesh = mesh2(64,48, newImg)
    newMesh.setHandlesOrg(pre.handlesAvg)
    newMesh.setHandlesDfm(newHandles)
    #newMesh.setHandlesDfm(handles)
    newMesh.applyHandles()

    newEye = newMesh.deform()

    #cv2.imwrite('output/'+ name + variation + '_projected_'+str(contRate)+'.png', newEye)


    #plt.imshow(cv2.cvtColor(newEye, cv2.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
    #plt.show()

    return newEye

def project_addDetail(inputImg, refImg, inputHandles, refHandles): # 投影した後, ほかの目データの詳細を加える
    orgImg = [0,0]
    prjImg = [0,0]
    hands = [refHandles, inputHandles]
    newHands = [0,0]
    texes = [refImg, inputImg]
    blur = 5
    diffScale = 1


    for i in range(2):
        eyeMesh = mesh2(64,48, texes[i]) # 入力と, 詳細を移植するリファレンス画像をメッシュ化
        eyeMesh.setHandlesOrg(hands[i])
        eyeMesh.setHandlesDfm(pre.handlesAvg)
        eyeMesh.applyHandles()

        newImg = eyeMesh.deform()

        orgImg[i] = newImg

        if i == 0:
            texB = cv2.GaussianBlur(texes[i], (blur, blur), blur) # リファレンスの詳細をブラーでぼかす
            eyeMesh2 = mesh2(64,48, texB)
            #print(handles)
            for j in range(len(hands[i])):
                rand = [[random.uniform(-1, 1), random.uniform(-1, 1)]]
                hands[i][j] = hands[i][j] + rand
            #print(handles)
            eyeMesh2.setHandlesOrg(hands[i])
            eyeMesh2.setHandlesDfm(pre.handlesAvg)
            eyeMesh2.applyHandles()

            newImg = eyeMesh2.deform()
            #plt.imshow(cv2.cvtColor(newImg, cv2.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
            #plt.show()

        avgEyeData = pre2.avgEyeData

        eyeVec = newImg.reshape(48*64*3)
        handleVec = hands[i].reshape(pre.H*1*2) * pre2.handCoeff

        eyeData = np.append(eyeVec, handleVec)

        eyeDataCenter = eyeData - avgEyeData

        if D != 0: # 入力, リファレンスを空間に投影
            x = P @ eyeDataCenter
            p = A @ x

            p = avgEyeData + p
            newImg, newHandles = np.split(p, [48*64*3])
        else:
            p = avgEyeData
            newImg, newHandles = np.split(p, [48*64*3])


        # 画像とハンドルに復元
        newImg = newImg.reshape(48,64,3)
        newImg = np.clip(newImg, 0, 255)
        newImg = newImg.astype(np.uint8)

        prjImg[i] = newImg

        newHandles = newHandles / pre2.handCoeff
        newHandles = newHandles.reshape(pre.H,1,2)
        newHands[i] = newHandles


        newMesh = mesh2(64,48, newImg)
        newMesh.setHandlesOrg(pre.handlesAvg)
        newMesh.setHandlesDfm(newHandles)
        #newMesh.setHandlesDfm(handles)
        newMesh.applyHandles()

        newEye = newMesh.deform()

    diff = orgImg[0]/1 - prjImg[0]/1 # リファレンスの元画像と投影画像の差分から詳細を抽出
    diff = diff - np.sum(diff)/(48*64*3) # 平均を0に（白飛び防止）
    newImg = diff*diffScale + prjImg[1]/1 # 入力の投影画像にリファレンスの詳細を足す
    newImg = np.clip(newImg, 0, 255) # 0-255に収める

    #メッシュ化して投影後のハンドルの形にして返す
    newMesh = mesh2(64,48, newImg)
    newMesh.setHandlesOrg(pre.handlesAvg)
    newMesh.setHandlesDfm(newHands[1])
    #newMesh.setHandlesDfm(handles)
    newMesh.applyHandles()
    newEye = newMesh.deform()

    return newEye

'''
texes = [0,0]
hands = [0,0]
names = ['rough_eye1', '025']
for i in range(2):
    tex = cv2.imread('data_eyes/'+ names[i] + '.png')

    texes[i] = tex
    handles = DMesh.detectP('data_eyes_p/'+ names[i] +'_p.png')
    handles2 = DMesh.detectP('data_eyes_p2/'+ names[i] +'_p2.png')
    handles3 = DMesh.detectP('data_eyes_p3/'+ names[i] +'_p3.png')

    handles = np.append(handles, handles2, axis=0)
    handles = np.append(handles, handles3, axis=0)
    handles = np.append(handles, np.array([[[32.5, 24.5]]]), axis=0)

    hands[i] = handles

newEye = project_addDetail(texes[0], texes[1], hands[0], hands[1])

plt.imshow(cv2.cvtColor(newEye, cv2.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
plt.show()
'''