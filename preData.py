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

# データの前処理（ハンドルによる目画像の変形, ベクトル化, 平均値）

# メッシュの頂点位置を画像中の点の位置から決定
img = cv2.imread('data/eye_mesh5.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hues = [0]
points = []

hsv_min = np.array([0,64,0])
hsv_max = np.array([0,255,255])
mask = cv2.inRange(hsv, hsv_min, hsv_max)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE ) 


for cnt in contours:
    M = cv2.moments(cnt)
    cx = M['m10']/M['m00']
    cy = M['m01']/M['m00']

    points.append(np.array([cx, cy]))

points = np.array(points)

points = points.astype(np.uint8)  # predata メッシュの頂点

img = np.zeros((48, 64, 3), np.uint8)
for p in points:
    cv2.drawMarker(img, tuple(p), (255, 255, 255), markerType=cv2.MARKER_SQUARE, markerSize=1)
    
rect = (0, 0, 64, 48)

subdiv = cv2.Subdiv2D(rect)

for p in points:
    subdiv.insert((p[0], p[1]))
    
triangles = subdiv.getTriangleList() # 三角形分割

pols = triangles.reshape(-1, 3, 2)

triangles_ind = np.zeros((len(triangles), 3))  # predata　メッシュを三角形分割するときの三角形の頂点インデックス

for j in range(len(pols)):
    for i in range(3):
        for p in range(len(points)):
            if pols[j][i][0] == points[p][0] and pols[j][i][1] == points[p][1]:
                triangles_ind[j][i] = p


# ハンドル平均
N = 143    # データの数
n = N - 3
H = 13     # ハンドルの数  # predata
handlesArr = []   # predata　各データのハンドルを格納
handlesAvg = np.zeros((H,1,2))

dummy = DMesh.detectP('data_eyes_p/001_p.png')
dummy2 = DMesh.detectP('data_eyes_p2/001_p2.png')
dummy3 = DMesh.detectP('data_eyes_p3/001_p3.png')

dummy = np.append(dummy, dummy2, axis=0)
dummy = np.append(dummy, dummy3, axis=0)
dummy = np.append(dummy, np.array([[[32.5, 24.5]]]), axis=0)

for i in range(N):
    
    if i == 84 or i == 122 or i == 123: # ベクトル化できそうにないものを除外
        handlesArr.append(dummy)
        continue
    
    handles = DMesh.detectP('data_eyes_p/'+ str(i+1).zfill(3)+ '_p.png')
    handles2 = DMesh.detectP('data_eyes_p2/'+ str(i+1).zfill(3)+ '_p2.png')
    handles3 = DMesh.detectP('data_eyes_p3/'+ str(i+1).zfill(3)+ '_p3.png')
    
    handles = np.append(handles, handles2, axis=0)
    handles = np.append(handles, handles3, axis=0)
    handles = np.append(handles, np.array([[[32.5, 24.5]]]), axis=0) # 中央の点（瞳孔位置）を追加
    handlesAvg += handles
    handlesArr.append(handles)
    
handlesAvg = handlesAvg / n   # predata　ハンドルの平均位置