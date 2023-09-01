# meshクラスの汎用性を高めたmesh2クラスの定義

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from ipywidgets import FloatSlider
import copy
import math
import pywt
import os

import DmeshLib as DMesh
import preData as pre

class mesh2: #meshクラスのメッシュの形状を任意にしたもの
    def __init__(self, width, height, img):
        self.width = width
        self.height = height
        self.vartices = np.copy(pre.points.astype(np.float64))
        self.varticesDfm = np.copy(self.vartices)
        self.triangles = np.copy(pre.triangles_ind.astype(np.int64))
                    
        self.texture = img
        self.image = cv2.resize(self.texture, (width, height))
        
    def deform(self, whiteback=False, outputMask=False):
        maskSum = np.zeros_like(self.image)
        newImg = np.zeros_like(self.image)
        for triangle in self.triangles:
            trianglePosOrg = np.array([self.vartices[triangle[0]], self.vartices[triangle[1]], self.vartices[triangle[2]]], dtype=np.float32)
            trianglePosDfm = np.array([self.varticesDfm[triangle[0]], self.varticesDfm[triangle[1]], self.varticesDfm[triangle[2]]], dtype=np.float32)
            affinMat = cv2.getAffineTransform(trianglePosOrg, trianglePosDfm)
            affinImg = cv2.warpAffine(self.image, affinMat, (self.width, self.height))

            black = np.zeros_like(self.image)
            mask = cv2.fillConvexPoly(black, trianglePosDfm.astype(np.int32), (255, 255, 255))        
            mask = mask / 255

            maskedImg = affinImg * mask
            newImg = newImg + maskedImg * (1 - maskSum)
            maskSum = np.clip(maskSum + mask, 0, 1) 
        
        if whiteback:
            white = np.full_like(self.image, 255)
            newImg = (newImg + white * (1 - maskSum)).astype(np.uint8)
        newImg = (newImg).astype(np.uint8)
        
        if outputMask:
            return newImg, maskSum
        else:
            return newImg
    
    def setHandlesOrg(self, handles, handlesAvg = pre.handlesAvg):
        self.handlesOrg = handles
        self.weightsArr = []
        self.AsArr = []
        
        tWArr = []
        tAArr = []
        
        for v in self.vartices:
            tWArr.append(DMesh.calcWeights(np.array([v]), handlesAvg))
            tAArr.append(DMesh.A_j(np.array([v]), handlesAvg))
        tWArr = np.array(tWArr)
        tAArr = np.array(tAArr)
        
        i = 0
        for w, A in zip(tWArr, tAArr):
            self.vartices[i] = DMesh.mls(w, A, handles)[0]
            i+=1
            
        for v in self.vartices:
            self.weightsArr.append(DMesh.calcWeights(np.array([v]), handles))
            self.AsArr.append(DMesh.A_j(np.array([v]), handles))
        self.weightsArr = np.array(self.weightsArr)
        self.AsArr = np.array(self.AsArr)
        
    def setHandlesDfm(self, handles):
        self.handlesDfm = handles
        
    def applyHandles(self):
        i = 0
        for w, A in zip(self.weightsArr, self.AsArr):
            self.varticesDfm[i] = DMesh.mls(w, A, self.handlesDfm)[0]
            i+=1