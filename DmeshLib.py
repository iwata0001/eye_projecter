import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from ipywidgets import FloatSlider
import copy
import math
import pywt
import os

alpha = 0.5
def calcWeights(v, handles): #moving least squares
    weights = []
    for handle in handles:
        if np.linalg.norm(v - handle) != 0:
            weights.append(1/(np.linalg.norm(v - handle)**(2*alpha)))
        else:
            weights.append(1/(0.00001**(2*alpha)))
    return np.array(weights)

def centroid(weights, handles): #moving least squares
    sumW = 0
    sumWH = 0
    for w, h in zip(weights, handles):
        sumW += w
        sumWH += w * h
    return sumWH / sumW

def A_j(v, handlesOrg): #moving least squares
    A = []
    weights = calcWeights(v, handlesOrg)
    centOrg = centroid(weights, handlesOrg)
    Bmat = [[0, 0], [0, 0]]
    for hO, w in zip(handlesOrg, weights):
        Bmat += (hO - centOrg).T @ (w * (hO - centOrg))
    Bmat = (v - centOrg) @ np.linalg.inv(Bmat)
    
    for hO, w in zip(handlesOrg, weights):
        A.append(Bmat @ (w * (hO - centOrg).T))
        
    return np.array(A)

def mls(weights, A, handlesDfm): #moving least squares法
    centDfm = centroid(weights, handlesDfm)
    newPos = np.copy(centDfm)
    for Aj, hD in zip(A, handlesDfm):
        newPos += Aj * hD
        
    return newPos

class mesh: #三角形に分割された長方形のメッシュ
    def __init__(self, divX: int, divY: int, width, height, img):
        self.divX = divX
        self.divY = divY
        self.width = width
        self.height = height
        self.vartices = np.zeros(((divY+1)*(divX+1), 2))
        meshW = width / divX
        meshH = height / divY
        for i in range(divY+1):
            for j in range(divX+1):
                self.vartices[i*(divY+1) + j] = np.array([meshW*j, meshH*i])
        self.varticesDfm = np.copy(self.vartices)
        self.triangles = np.zeros((divY*divX*2, 3))
        for i in range(divY):
            for j in range(divX):
                self.triangles[(i*(divY) + j)*2] = [i*(divY)+j+i, i*(divY)+j+i+divY+1, i*(divY)+j+i+divY+2]
                self.triangles[(i*(divY) + j)*2+1] = [i*(divY)+j+i, i*(divY)+j+i+divY+2, i*(divY)+j+i+1]
        self.triangles = self.triangles.astype(np.int64)
        self.adjacentTriangles = []
        for i in range(len(self.vartices)):
            self.adjacentTriangles.append([])
        for i in range(len(self.triangles)):
            for j in range(3):
                if not(i in self.adjacentTriangles[self.triangles[i][j]]):
                    self.adjacentTriangles[self.triangles[i][j]].append(i)
                    
        self.texture = img
        self.image = cv2.resize(self.texture, (width, height))
        
    def deform(self, w = None, h = None): #メッシュの変形処理
        if w == None:
            w = self.width
        if h == None:
            h = self.height
        maskSum = np.zeros((h, w, 3))
        newImg = np.zeros((h, w, 3))
        for triangle in self.triangles:
            trianglePosOrg = np.array([self.vartices[triangle[0]], self.vartices[triangle[1]], self.vartices[triangle[2]]], dtype=np.float32)
            trianglePosDfm = np.array([self.varticesDfm[triangle[0]], self.varticesDfm[triangle[1]], self.varticesDfm[triangle[2]]], dtype=np.float32)
            affinMat = cv2.getAffineTransform(trianglePosOrg, trianglePosDfm)
            affinImg = cv2.warpAffine(self.image, affinMat, (w, h), borderValue=(255, 255, 255))

            black = np.zeros((h, w, 3))
            mask = cv2.fillConvexPoly(black, trianglePosDfm.astype(np.int32), (255, 255, 255))        
            mask = mask / 255

            maskedImg = affinImg * mask
            newImg = newImg + maskedImg * (1 - maskSum)
            maskSum = np.clip(maskSum + mask, 0, 1) 

        white = np.full((h, w, 3), 255)
        white = white*(1-maskSum)

        newImg = newImg + white
            
        #newImg = (newImg + self.image * (1 - maskSum)).astype(np.uint8)
        newImg = (newImg).astype(np.uint8)
        
        return newImg
    
    def setHandlesOrg(self, handles): # 変形に用いるハンドルの初期位置を設定
        self.handlesOrg = handles
        self.weightsArr = []
        self.AsArr = []
        for v in self.vartices:
            self.weightsArr.append(calcWeights(np.array([v]), handles))
            self.AsArr.append(A_j(np.array([v]), handles))
        self.weightsArr = np.array(self.weightsArr)
        self.AsArr = np.array(self.AsArr)
        
    def setHandlesDfm(self, handles): # ハンドルの変形後の位置を設定
        self.handlesDfm = handles
        
    def applyHandles(self): # ハンドルの位置からメッシュの各頂点の位置を設定
        i = 0
        for w, A in zip(self.weightsArr, self.AsArr):
            self.varticesDfm[i] = mls(w, A, self.handlesDfm)[0]
            i+=1

def detectP(url): # hues = [0, 45, 90, 135]の色の点の位置を返す関数
    img = cv2.imread(url)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hues = [0, 45, 90, 135]
    points = []
    
    for hue in hues:
        hsv_min = np.array([hue,64,0])
        hsv_max = np.array([hue,255,255])
        mask = cv2.inRange(hsv, hsv_min, hsv_max)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE ) 

        cnt = contours[0]
        M = cv2.moments(cnt)
        cx = M['m10']/M['m00']
        cy = M['m01']/M['m00']
        
        points.append([[cx, cy]])
        
    return np.array(points)

def freqDatas(gray, level): # モノクロ画像をウェーブレット変換する
    datas = []
    shapes = []
    coeffs = pywt.wavedec2(gray, 'bior1.3', level=level)
    for i in range(level+1):
        datas.append(np.array(coeffs[i]).reshape(np.prod(np.array(coeffs[i]).shape)))
        shapes.append(np.array(coeffs[i]).shape)
        
    return datas, shapes

def recovFreqDatas(datas, shapes): # 逆ウェーブレット
    coeffs = []
    for data, shape in zip(datas, shapes):
        coeff = data.reshape(shape)
        if len(shape) == 2:
            coeffs.append(coeff)
        else:
            temp = []
            for i in range(shape[0]):
                temp.append(coeff[i])
            coeffs.append(tuple(temp))
            
    coeffs = tuple(coeffs)
    img = pywt.waverec2(coeffs,'bior1.3')
    return img

