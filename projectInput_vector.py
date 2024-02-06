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
from utlLib import calcHandleDiff

# 主成分の変化による画像の変化量 i番めの主成分が+-に振れたときに画像がどれくらい変化するかを示す
MPdiffs = [12532.252, 5648.708, 5056.236, 3802.8079, 2766.4272, 5852.853, 3787.906, 3973.0425, 3225.1687, 2532.366, 3392.5527, 2676.435, 2152.957, 3320.0884, 2275.8564, 2331.2412, 2525.0254, 2117.9512, 1736.1089, 1977.7578, 1823.0757, 1475.5413, 1559.0629, 1340.4797, 1418.5654, 983.4246, 1375.1382, 1186.6836, 1164.0645, 1151.7808, 1333.0532, 1057.6473, 993.62213, 930.4961, 1089.5468, 1109.5305, 978.5346, 963.71313, 1084.7765, 941.3655, 841.4113, 991.9471, 835.6237, 981.64404, 1017.64777, 879.37195, 975.25024, 823.7585, 1024.5712, 839.90356, 780.5626, 788.0761, 714.30316, 766.477, 713.1704, 720.0049, 677.66956, 712.28925, 664.97516, 681.1755, 561.4161, 671.5043, 633.45166, 600.64465, 644.67975, 585.8285, 643.76624, 541.5801, 649.7792, 517.7702, 538.6353, 587.26056, 572.3504, 518.7032, 492.10364, 540.3536, 547.58746, 566.7389, 466.06546, 494.00708, 529.57153, 443.5617, 468.54028, 486.79666, 431.39658, 441.53595, 458.9902, 419.60577, 483.81607, 383.45404, 407.81735, 384.3579, 552.59296, 427.91705, 442.94922, 435.564, 337.36627, 364.6135, 391.67587, 5.7445626]

LabMode = pre2.LabMode

# 主成分分析 固有ベクトルを読み込んで固有値順に
eyeCoeff = pre2.eyeCoeff
handCoeff = pre2.handCoeff_v1
vecCoeff = pre2.vecCoeff
if LabMode:
    eigVal = np.load('saves/100dataLab_eigval.npy')
    eigVec = np.load('saves/100dataLab_eigvec.npy')
else:
    eigVal = np.load('saves/100data_eigval.npy')
    eigVec = np.load('saves/100data_eigvec.npy')
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

handlesAdd = handles + np.array([[3,4]])

#print(calcHandleDiff(handles, handlesAdd))

#print(handles.shape)

#累積寄与率がcontRateになるまで固有ベクトルを並べる
contRate = 0.75
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

def project_eigSpace2(texture, handles, contRate=0.95):
    #累積寄与率がcontRateになるまで固有ベクトルを並べる
    temp = 0
    D = 0

    for n in range(len(eigVal)):
        temp += eigVal[indices[n]]
        if temp / eigValSum > contRate:
            D = n
            break
    print("model dimension:", D)
    print("contRate:",temp / eigValSum)

    A = []
    for n in range(D):
        A.append(eigVec[indices[n]])

    # 並べた固有ベクトルで張られる空間に点を正射影する行列Pを求める
    A = np.array(A)
    A = A.T
    Q = A @ np.linalg.inv(A.T @ A) @ A.T - np.identity(9290)

    print(A.shape, Q.shape)

    Qi, Qj = np.split(Q,[48*64*3+13*1*2],axis=1)

    tex = texture
    eyeMesh = mesh2(64,48, tex)
    eyeMesh.setHandlesOrg(handles)
    eyeMesh.setHandlesDfm(pre.handlesAvg)
    eyeMesh.applyHandles()
    img, imgMask= eyeMesh.deform(outputMask=True)

    avgImgData, avgHandleData, avgVectorData = np.split(pre2.avgdata_v1, [48*64*3, 48*64*3+13*1*2])

    eyeVec = img.reshape(48*64*3) * eyeCoeff - avgImgData
    handleVec = handles.reshape(pre.H*1*2) * handCoeff - avgHandleData
    knownVec = np.append(eyeVec, handleVec)
    beta = Qi @ knownVec * (-1)

    newVectorVec = np.linalg.inv(Qj.T @ Qj) @ Qj.T @ beta

    completeVec = np.append(knownVec, newVectorVec)

    c = np.linalg.inv(A.T @ A) @ A.T @ completeVec

    newDataVec = A @ c

    print(newDataVec.shape)

    newDataVec = newDataVec+pre2.avgdata_v1
    newImg, newHandles, newVector = np.split(newDataVec, [48*64*3, 48*64*3+13*1*2])

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
    newEye, newEyeMask = newMesh.deform(outputMask=True)

    N = 5
    newVectorN = newVector/vecCoeff
    UOx, UOy, UIx, UIy, LOx, LOy, LIx, LIy, ppl = np.split(newVectorN, [N, 2*N, 3*N, 4*N, 5*N, 6*N, 7*N, 8*N])
    pplXY = []
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
    
    with open('json_data/vec_eigSpace.json', 'w') as f:
        json.dump(vecdata, f)

    return newEye, newHandles


i = 120
tex = cv2.imread('data_eyes_test/'+str(i)+'.png')
handle = pre.handlesArr[i-1]
newEye, __newHandle = project_eigSpace2(tex, handle)
cv2.imwrite('C:/pics/eigSpace/eigSpaceProject'+str(i)+'.png', newEye)


    

def project_eigSpace(texture, handles, contRate=0.95):
    #累積寄与率がcontRateになるまで固有ベクトルを並べる
    temp = 0
    D = 0

    for n in range(len(eigVal)):
        temp += eigVal[indices[n]]
        if temp / eigValSum > contRate:
            D = n
            break
    print("model dimension:", D)
    print("contRate:",temp / eigValSum)

    A = []
    for n in range(D):
        A.append(eigVec[indices[n]])

    # 並べた固有ベクトルで張られる空間に点を正射影する行列Pを求める
    A = np.array(A)
    A = A.T
    P = np.linalg.pinv(A.T @ A) @ A.T

    print(A.shape, P.shape)

    Pi, Pj = np.split(P,[48*64*3+13*1*2],axis=1)

    print(Pi.shape, Pj.shape)

    tex = texture
    eyeMesh = mesh2(64,48, tex)
    eyeMesh.setHandlesOrg(handles)
    eyeMesh.setHandlesDfm(pre.handlesAvg)
    eyeMesh.applyHandles()
    img, imgMask= eyeMesh.deform(outputMask=True)

    avgImgData, avgHandleData, avgVectorData = np.split(pre2.avgdata_v1, [48*64*3, 48*64*3+13*1*2])

    eyeVec = img.reshape(48*64*3) * eyeCoeff - avgImgData
    handleVec = handles.reshape(pre.H*1*2) * handCoeff - avgHandleData
    knownVec = np.append(eyeVec, handleVec)
    alpha = Pi @ knownVec * (-1)

    print(alpha.shape)

    # Pjを構成する列ベクトルで張られる空間にalphaを投影
    alpha_proj = Pj @ np.linalg.inv(Pj.T @ Pj) @ Pj.T @ alpha

    print(alpha_proj.shape)

    c = alpha_proj - alpha

    for j in range(c.shape[0]):
        if j < 6:
            coeff = 5
        elif j < 20:
            coeff = 3
        else:
            coeff = 0.5
        newxElm = np.clip(np.array([c[j]]), (-1*coeff)*np.sqrt(eigVal[indices[j]]), coeff*np.sqrt(eigVal[indices[j]]))
        if c[j] != newxElm[0]: # 主成分方向の分散に対して離れているものを無視（平均から離れすぎないようにする）
            c[j] = 0
    newDataVec = A @ c

    print(newDataVec.shape)

    newDataVec = newDataVec+pre2.avgdata_v1
    newImg, newHandles, newVector = np.split(newDataVec, [48*64*3, 48*64*3+13*1*2])

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

    newEye, newEyeMask = newMesh.deform(outputMask=True)

    return newEye, newHandles

for i in range(110,121):
    tex = cv2.imread('data_eyes_test/'+str(i)+'.png')
    handle = pre.handlesArr[i-1]
    #newEye, newHandles = project_eigSpace(tex,handle,contRate=0.95)
    #cv2.imwrite('C:/pics/eigSpace/eigSpaceProject'+str(i)+'.png', newEye)


def project_withVector(texture, handles, testNumber=None, outputErrors=False, contRate=0.75, refs=None, isSelectEig=False): # refs = [contRate=double, img=(48,64,3), handle=(13,1,2)]

    #累積寄与率がcontRateになるまで固有ベクトルを並べる
    temp = 0
    D = 0
    D_ref = 0
    refexist = False
    refImgDetail = np.zeros_like(texture)

    for n in range(len(eigVal)):
        temp += eigVal[indices[n]]
        if temp / eigValSum > contRate:
            D = n
            break
    print("model dimension:", D)
    print("contRate:",temp / eigValSum)

    if refs!=None and contRate<refs[0]:
        n = 0
        temp = 0
        for n in range(len(eigVal)):
            temp += eigVal[indices[n]]
            if temp / eigValSum > refs[0]:
                D_ref = n
                break

    A = []
    for n in range(D):
        A.append(eigVec[indices[n]])

    if D < D_ref:
        refexist = True
        A_ref = []
        for n in range(D_ref):
            A_ref.append(eigVec[indices[n]])


    # 並べた固有ベクトルで張られる空間に点を正射影する行列Pを求める
    A = np.array(A)
    A = A.T
    P = np.linalg.inv(A.T @ A) @ A.T

    #texture = cv2.imread('data_eyes/001.png')
    tex = texture
    eyeMesh = mesh2(64,48, tex)
    eyeMesh.setHandlesOrg(handles)
    eyeMesh.setHandlesDfm(pre.handlesAvg)
    eyeMesh.applyHandles()

    img, imgMask= eyeMesh.deform(outputMask=True)

    if refexist:
        print("refexist is True")
        A_ref = np.array(A_ref)
        A_ref = A_ref.T
        P_ref = np.linalg.inv(A_ref.T @ A_ref) @ A_ref.T

        tex_ref = refs[1]
        eyeMesh_ref = mesh2(64,48, tex_ref)
        eyeMesh_ref.setHandlesOrg(refs[2])
        eyeMesh_ref.setHandlesDfm(pre.handlesAvg)
        eyeMesh_ref.applyHandles()

        img_ref, imgMask_ref= eyeMesh_ref.deform(outputMask=True)

    cv2.imwrite('temp_img/shapeNormalizedRGBSketch.png',img)
    if LabMode:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # Labに変換　固有値データを置き換えてください
    avgEyeData = pre2.avgdata_v1

    avgImgData, avgHandleData, avgVectorData = np.split(avgEyeData, [48*64*3, 48*64*3+13*1*2])

    #print(avgEyeData.shape, avgImgData.shape, avgHandleData.shape, avgVectorData.shape)
    vectorTemp = avgVectorData
    vectorTemp = np.array(vectorTemp)

    #初期値ベクターデータを保存
    N = 5
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
    eyeVec = img.reshape(48*64*3) * eyeCoeff
    handleVec = handles.reshape(pre.H*1*2) * handCoeff
    for i in range(100):
        # 画像(eyeVec)、特徴点(handleVec)、ベクターデータ(vectorTemp)をくっつける
        eyeData = np.append(eyeVec, handleVec)
        eyeData = np.append(eyeData, vectorTemp)

        # 平均が０になるように正規化
        eyeDataCenter = eyeData - avgEyeData

        # 投影
        if D != 0:
            x = P @ eyeDataCenter

            if i == 99 and isSelectEig:
                print("x.size",x.size)
                temp = 0
                for j in range(x.size):
                    temp += eigVal[indices[j]]
                    if j < 6:
                        c = 5
                    elif j < 20:
                        c = 3
                    else:
                        c = 0.5
                    print(j, "std:", np.sqrt(eigVal[indices[j]]), x[j], end="")
                    newxElm = np.clip(np.array([x[j]]), (-1*c)*np.sqrt(eigVal[indices[j]]), c*np.sqrt(eigVal[indices[j]]))
                    if x[j] != newxElm[0]: # 主成分方向の分散に対して離れているものを無視（平均から離れすぎないようにする）
                        x[j] = 0
                        temp -= eigVal[indices[j]]
                        print(" ignore", end="")
                    print("")
                print("actual contRate:",temp/eigValSum)
            p = A @ x
                    
            p = avgEyeData + p
            newImg, newHandles, newVector = np.split(p, [48*64*3, 48*64*3+13*1*2])
        else:
            p = avgEyeData
            newImg, newHandles, newVector = np.split(p, [48*64*3, 48*64*3+13*1*2])

        vectorTemp = newVector

        #入力と出力の二乗誤差(特徴点)
        ioDiff = np.mean((handleVec-newHandles)**2 / (handCoeff**2))
        #normDiff = np.linalg.norm(np.array(vectorTemp) - np.array(newVector))
        #print("ioDiff:",ioDiff)
        diffs.append(ioDiff)
        xs.append(i)

        
        # 収束過程の画像保存
        #############################################################################
        '''
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
        for pos in newHandlesTemp:
            p = (int(pos[0][0]),int(pos[0][1]))
            cv2.circle(newEye, p, 1, (0,0,255), thickness=-1)
        cv2.imwrite('output/fitting_img/'+'img_itr'+str(i+1)+'.png', newEye)
        '''
        #############################################################################


        #収束過程のベクターデータを保存
        #############################################################################
        '''
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
        '''
        #############################################################################

    if refexist: # refsがあればそれを投影
        vectorTemp = avgVectorData
        vectorTemp = np.array(vectorTemp)
        eyeVec_ref = img_ref.reshape(48*64*3) * eyeCoeff
        handleVec_ref = refs[2].reshape(pre.H*1*2) * handCoeff
        for i in range(100):
            # 画像(eyeVec)、特徴点(handleVec)、ベクターデータ(vectorTemp)をくっつける
            eyeData_ref = np.append(eyeVec_ref, handleVec_ref)
            eyeData_ref = np.append(eyeData_ref, vectorTemp)

            # 平均が０になるように正規化
            eyeDataCenter_ref = eyeData_ref - avgEyeData

            # 投影
            if D_ref != 0:
                x_ref = P_ref @ eyeDataCenter_ref
                p_ref = A_ref @ x_ref

                p_ref = avgEyeData + p_ref
                newImg_ref, newHandles_ref, newVector_ref = np.split(p_ref, [48*64*3, 48*64*3+13*1*2])
            else:
                p_ref = avgEyeData
                newImg_ref, newHandles_ref, newVector_ref = np.split(p_ref, [48*64*3, 48*64*3+13*1*2])

            vectorTemp = newVector_ref

        newImg_ref = newImg_ref.reshape(48,64,3)
        refImgDetail = img_ref - newImg_ref
        refImgDetail = refImgDetail - np.mean(refImgDetail)
        #mean = np.mean(refImgDetail)
        for i in range(x.shape[0]): # パラメータの係数の列x_refの前半をxにして固有値の線形和をる
            x_ref[i] = x[i]

        p_ref = A_ref @ x_ref
        p_ref = avgEyeData + p_ref
        newImg, newHandles, newVector = np.split(p_ref, [48*64*3, 48*64*3+13*1*2])
        

    #plt.plot(xs, diffs)
    #plt.show()
    newImg = newImg / eyeCoeff
    newImg = newImg.reshape(48,64,3)
    #newImg = newImg + refImgDetail
    newImg = np.clip(newImg, 0, 255)
    newImg = newImg.astype(np.uint8)
    if LabMode:
        newImg = cv2.cvtColor(newImg, cv2.COLOR_LAB2BGR) # LabからBGRに戻す　固有値データ置き換え
    cv2.imwrite('temp_img/shapeNormalizedOutputImg.png', newImg)

    newHandles = newHandles / handCoeff
    newHandles = newHandles.reshape(pre.H,1,2)

    newMesh = mesh2(64,48, newImg)
    newMesh.setHandlesOrg(pre.handlesAvg)
    newMesh.setHandlesDfm(newHandles)
    #newMesh.setHandlesDfm(handles)
    newMesh.applyHandles()

    newEye, newEyeMask = newMesh.deform(outputMask=True)

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

    if testNumber != None:
        with open('test_result/json/test'+str(testNumber+1)+'.json', 'w') as f:
            json.dump(vecdata, f)
    
    with open('json_data/avg_v2.json', 'w') as f:
        json.dump(vecdata, f)
    #print("saved.")

    # 画像部分の誤差計算
    if testNumber != None:
        newImgLab = cv2.cvtColor(newImg, cv2.COLOR_BGR2LAB)
        imgLab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) 
        if LabMode:
            imgLab = img
        
        #cv2.imshow("Image1", img)
        #cv2.imshow("Image2", newImg)
        #cv2.waitKey()
        newImg = newImg.astype(np.float32)
        newImgLab = newImgLab.astype(np.float32)
        img = img.astype(np.float32)
        imgLab = imgLab.astype(np.float32)

        imgDiff = newImg - img
        imgDiffLab = newImgLab - imgLab

        deltaE = 0
        E = 0
        deltaELab = 0
        ELab = 0

        mseRGB = 0
        mseL = 0
        mseab = 0
        pixcelNum = 0

        for i in range(48):
            for j in range(64):
                if np.any(newEyeMask[i][j] == 1):
                    deltaE += np.sqrt(imgDiff[i][j][0]**2 + imgDiff[i][j][1]**2 + imgDiff[i][j][2]**2)
                    E += np.sqrt(img[i][j][0]**2 + img[i][j][1]**2 + img[i][j][2]**2)

                    mseRGB += (imgDiff[i][j][0]**2 + imgDiff[i][j][1]**2 + imgDiff[i][j][2]**2)

                    deltaELab += np.sqrt(imgDiffLab[i][j][0]**2 + imgDiffLab[i][j][1]**2 + imgDiffLab[i][j][2]**2)
                    ELab += np.sqrt(imgLab[i][j][0]**2 + imgLab[i][j][1]**2 + imgLab[i][j][2]**2)

                    mseL += imgDiffLab[i][j][0]**2
                    mseab += (imgDiffLab[i][j][1]**2 + imgDiffLab[i][j][2]**2)

                    pixcelNum += 1

        imgErrRGB = deltaE/E
        imgErrLab = deltaELab/ELab

        mseRGB = mseRGB/(pixcelNum*3)
        mseL = mseL/pixcelNum
        mseab = mseab/(pixcelNum*2)

        psnrRGB = 10*np.log10(255**2/mseRGB)
        psnrL = 10*np.log10(255**2/mseL)
        psnrab = 10*np.log10(255**2/mseab)

    # 誤差計算
    if testNumber != None:

        # アンカーポイント誤差計算
        testHandles = pre.handlesArr[testNumber]
        anchorErrPixcel, anchorErrRatio, boundingBoxLen = calcHandleDiff(testHandles, newHandles)

        # vector(まつ毛，瞳)誤差計算
        trueVector = pre2.testVectors[testNumber]
        UOxT, UOyT, UIxT, UIyT, LOxT, LOyT, LIxT, LIyT, pplT = np.split(trueVector, [N, 2*N, 3*N, 4*N, 5*N, 6*N, 7*N, 8*N])
        UO = []
        UI = []
        LO = []
        LI = []
        UOT = []
        UIT = []
        LOT = []
        LIT = []
        for i in range(5):
            UO.append(np.array([UOx[i],UOy[i]]))
            UI.append(np.array([UIx[i],UIy[i]]))
            LO.append(np.array([LOx[i],LOy[i]]))
            LI.append(np.array([LIx[i],LIy[i]]))
            UOT.append(np.array([UOxT[i],UOyT[i]]))
            UIT.append(np.array([UIxT[i],UIyT[i]]))
            LOT.append(np.array([LOxT[i],LOyT[i]]))
            LIT.append(np.array([LIxT[i],LIyT[i]]))
        ppls = []
        pplsT = []
        for i in range(4):
            ppls.append(np.array([ppl[2*i], ppl[2*i+1]]))
            pplsT.append(np.array([pplT[2*i], pplT[2*i+1]]))
        eyelash = UO + UI + LO + LI + ppls
        eyelashTrue = UOT + UIT + LOT + LIT + pplsT
        eyelash = np.array(eyelash)
        eyelashTrue = np.array(eyelashTrue)
        eyelashErr = eyelashTrue - eyelash
        num = 0
        errDist = 0
        for err in eyelashErr:
            num += 1
            errDist += np.linalg.norm(err, ord=2)
        outlErr = errDist/num/(10*boundingBoxLen)

        # アンカーポイント誤差計算
        testHandles = pre.handlesArr[testNumber]
        anchorErrPixcel, anchorErrRatio, boundingBoxLen = calcHandleDiff(testHandles, newHandles)


        print("imgErrorRGB",imgErrRGB)
        print("psnrRGB", psnrRGB)
        print("imgErrorLab",imgErrLab)
        print("psnrL,ab", psnrL,psnrab)
        print("outlineErr(pixel):",outlErr)
        print("handleDiff(pixel),(ratio)", anchorErrPixcel, anchorErrRatio)
        errors = [testNumber+1, imgErrRGB, psnrRGB, imgErrLab, psnrL, psnrab, outlErr, anchorErrPixcel, anchorErrRatio]
        errors = [testNumber+1, imgErrLab, anchorErrRatio, outlErr]
        errors = np.array(errors)
        
        resultImg = np.append(texture,newEye,axis=1)
        cv2.imwrite('test_result/'+'test'+str(testNumber+1)+'.png', resultImg)

    #plt.imshow(cv2.cvtColor(newEye, cv2.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
    #plt.show()

    return newEye, newHandles, newEyeMask, errors if outputErrors else None