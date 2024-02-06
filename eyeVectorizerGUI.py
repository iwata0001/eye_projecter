# eyeProjectorGUIが作ったデータのうちベクトル画像データを可視化・編集するGUI
# モデルのためのベクトル画像データ作成ツールも兼ねる

from distutils.cmd import Command
from statistics import variance
import tkinter
from tkinter import Variable, ttk
from PIL import Image, ImageDraw
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from ipywidgets import FloatSlider
import copy
import math
import pywt
import os
import io
from PIL import ImageTk
from scipy import interpolate
from tkinter import colorchooser
import json
import time

import DmeshLib
from utlLib import createOvalEZ

def LOinside(n):
    if n==4:
        return 0
    elif n==5:
        return 4
    elif n==6:
        return 5
    else:
        return None

def colorCord(R,G,B):
    return('#'+ format(R, '02x')+ format(G, '02x')+ format(B, '02x'))

def sortDrawOrder(canvas, order): # order = [tag0, tag1, tag2, ...] 上から
    for tag in order:
        canvas.tag_lower(tag)

def closs2D(u, v):
    return (u[0]*v[1] - u[1]*v[0])

def dot2D(u, v):
    return (u[0]*v[0] + u[1]*v[1])

def abs2D(v):
    return(np.sqrt(v[0]**2 + v[1]**2))

def signedAngle2D(u, v): # [-PI, PI] uからvへの符号付き角度
    closs = closs2D(u, v)
    dot = dot2D(u, v)
    absUV = abs2D(u)*abs2D(v)
    cosTheta = dot/absUV
    sinTheta = closs/absUV
    theta = np.arcsin(sinTheta)
    if cosTheta < 0:
        theta = theta/abs(theta) * math.pi - theta
    return(theta)

def isInside(p, polygon): # 頂点の集合polygonは反時計回りで与える
    angle = 0
    for i in range(len(polygon)):
        vecA = [polygon[i][0] - p[0], polygon[i][1] - p[1]]
        if i == len(polygon) - 1:
            vecB = [polygon[0][0] - p[0], polygon[0][1] - p[1]]
        else:
            vecB = [polygon[i+1][0] - p[0], polygon[i+1][1] - p[1]]

        angle = angle + signedAngle2D(vecA, vecB)
    
    tol = 0.1
    if 2*math.pi-tol < angle and angle < 2*math.pi+tol:
        return True
    elif (-1)*tol < angle and angle < tol:
        return False
    else:
        return None

def getNearest(p, Ps):
    lenSq = 9999999999999999
    minInd = -1
    for i in range(len(Ps)):
        #print((Ps[i][0] - p[0])**2 + (Ps[i][1] - p[1])**2)
        if lenSq > (Ps[i][0] - p[0])**2 + (Ps[i][1] - p[1])**2:
            lenSq = (Ps[i][0] - p[0])**2 + (Ps[i][1] - p[1])**2
            minInd = i
    return minInd
        

def spline(x,y,point,deg):
    tck,u = interpolate.splprep([x,y],k=deg,s=0) 
    u1 = np.linspace(0,1,num=point,endpoint=True) 
    spline = interpolate.splev(u1,tck)
    return spline[0],spline[1],u

def spline1D(x,y,point):
    f = interpolate.interp1d(x, y,kind="cubic") 
    X = np.linspace(x[0],x[-1],num=point,endpoint=True)
    Y = f(X)
    return X,Y


def map0to1(min, max, n):
    return (n-min)/(max-min)

def circle(canvas, x, y, rad, col1, col2, col3, col4, tag):
    colD = col2 - col1
    tex = np.zeros((100,100,3))
    for i in range(2*rad):
        for j in range(2*rad):
            if (i-rad)**2 + (j-rad)**2 <= (rad)**2:
                newCol = col1 + math.sqrt((i-rad)**2 + (j-rad)**2)/rad * colD + (col3 + j/(2*rad+1)*col4)
                newCol = np.clip(newCol, 0, 255)
                newCol = newCol.astype(np.uint8)
                recX = x-rad+i
                recY = y-rad+j
                canvas.create_rectangle(recX, recY, recX+1, recY+1, fill=colorCord(*newCol), width=0, tag = tag)

def pupil_mask(rad):
    tex = np.full((2*rad+1, 2*rad+1, 3), 0)
    for i in range(2*rad+1):
        for j in range(2*rad+1):
            if (i-rad)**2 + (j-rad)**2 <= (rad+0.4)**2:
                tex[j][i] = np.array([255,0,0])
    tex = tex.astype(np.uint8)

def circleTex(rad, col1, col2, col3, col4):
    tex = np.full((2*rad+1, 2*rad+1, 3), 255)
    colD = col2 - col1
    for i in range(2*rad+1):
        for j in range(2*rad+1):
            if (i-rad)**2 + (j-rad)**2 <= (rad+0.4)**2:
                newCol = col1 + math.sqrt((i-rad)**2 + (j-rad)**2)/rad * colD + (col3 + j/(2*rad+1)*col4)
                newCol = np.clip(newCol, 0, 255)
                newCol = newCol.astype(np.uint8)
                tex[j][i] = newCol
    tex = tex.astype(np.uint8)

    return tex

def circleTex_2(rad1, rad2, col1, col2, filter1 = np.array([255,0,0]), filter2 = np.array([0,255,0]), filter3 = np.array([0,0,255]), isGradation = False):
    tex = np.full((2*rad1+1, 2*rad1+1, 3), 255)
    colD = col2 - col1
    for i in range(2*rad1+1):
        for j in range(2*rad1+1):
            if (i-rad1)**2 + (j-rad1)**2 <= (rad1+0.4)**2:
                tex[j][i] = col1

                if isGradation:
                    distance = ((i-rad1)**2 + (j-rad1-rad1)**2)**(1/2)
                    if distance < rad1:
                        k = map0to1(0, rad1, distance)
                        newCol = k*filter2 + (1-k)*filter1
                        newCol = np.clip(newCol, 0, 255)
                        newCol = newCol.astype(np.uint8)
                        tex[j][i] = newCol
                    else:
                        k = map0to1(rad1, 2*rad1, distance)
                        newCol = k*filter3 + (1-k)*filter2
                        newCol = np.clip(newCol, 0, 255)
                        newCol = newCol.astype(np.uint8)
                        tex[j][i] = newCol
                else:
                    if (i-rad1)**2 + (j-rad1 -rad1)**2 <= (rad1/3 *2)**2:
                        tex[j][i] = filter1
                    elif (i-rad1)**2 + (j-rad1 -rad1)**2 <= (rad1/3 *2 *2)**2:
                        tex[j][i] = filter2
                    else:
                        tex[j][i] = filter3

            if (i-rad1)**2 + (j-rad1)**2 <= (rad2+0.4)**2:
                tex[j][i] = col2
    tex = tex.astype(np.uint8)

    return tex

tex = circleTex(50, np.array([0,0,0]), np.array([250,250,250]), np.array([-0,-0,-0])*70, np.array([0,0,0])*120)
#cv2.imshow("image", tex)
#cv2.waitKey()

def func(x):
    return math.sin(x)

class eyelash:
    def __init__(self, shapeX, shapeY, rad, N):
        self.shapeX = shapeX
        self.shapeY = shapeY
        self.rad = rad
        self.length = N

        radY = list(range(0,len(rad), 1))
        #print(len(rad), radY)

        self.shapeXs, self.shapeYs, u = spline(shapeX,shapeY,N,3)
        self.rads, radYs, u = spline(rad, radY, N, 3)

    def setShape(self, shapeX, shapeY):
        self.shapeX = shapeX
        self.shapeY = shapeY
        self.shapeXs, self.shapeYs, u = spline(shapeX,shapeY,self.length,3)
        self.shapeU = u

    def setRad(self, rad):
        self.rad = rad
        radY = list(range(0,len(rad), 1))
        self.radYs, self.rads = spline1D(self.shapeU, rad, self.length)
        #print(self.shapeU, self.radYs)

    def setLength(self, N):
        self.length = N
        self.shapeXs, self.shapeYs, u = spline(self.shapeX,self.shapeY,self.length,3)
        radY = list(range(0,len(self.rad), 1))
        self.rads, radYs, u = spline(self.rad, radY, self.length, 3)

class eyelash_v2:
    def __init__(self, shapeOuterX, shapeOuterY, shapeInnerX, shapeInnerY, N):
        self.shapeOuterX = shapeOuterX
        self.shapeOuterY = shapeOuterY
        self.shapeInnerX = shapeInnerX
        self.shapeInnerY = shapeInnerY

        self.length = N

        #print(len(rad), radY)

        self.shapeOuterXs, self.shapeOuterYs, u = spline(shapeOuterX,shapeOuterY,N,3)
        self.shapeInnerXs, self.shapeInnerYs, u = spline(shapeInnerX,shapeInnerY,N,3)

    def setShapeOuter(self, shapeX, shapeY):
        self.shapeOuterX = shapeX
        self.shapeOuterY = shapeY
        self.shapeOuterXs, self.shapeOuterYs, u = spline(shapeX,shapeY,self.length,3)

    def setHandleOuter(self, handleInd, handleX, handleY):
        self.shapeOuterX[handleInd] = handleX
        self.shapeOuterY[handleInd] = handleY
        self.shapeOuterXs, self.shapeOuterYs, u = spline(self.shapeOuterX,self.shapeOuterY,self.length,3)

    def setShapeInner(self, shapeX, shapeY):
        self.shapeInnerX = shapeX
        self.shapeInnerY = shapeY
        self.shapeInnerXs, self.shapeInnerYs, u = spline(shapeX,shapeY,self.length,3)

    def setHandleInner(self, handleInd, handleX, handleY):
        self.shapeInnerX[handleInd] = handleX
        self.shapeInnerY[handleInd] = handleY
        self.shapeInnerXs, self.shapeInnerYs, u = spline(self.shapeInnerX,self.shapeInnerY,self.length,3)

    def setLength(self, N):
        self.length = N
        self.shapeOuterXs, self.shapeOuterYs, u = spline(self.shapeOuterX,self.shapeOuterY,self.length,3)
        self.shapeInnerXs, self.shapeInnerYs, u = spline(self.shapeInnerX,self.shapeInnerY,self.length,3)

    def draw(self, i, canvas, color, tag, isUpper = True):
        #canvas.delete(tag)
        canvas.create_polygon(self.shapeOuterXs[i],self.shapeOuterYs[i],
        self.shapeOuterXs[i+1],self.shapeOuterYs[i+1],
        self.shapeInnerXs[i+1],self.shapeInnerYs[i+1],
        self.shapeInnerXs[i],self.shapeInnerYs[i], 
        fill=color, tag = tag)

class eye: # U:upper, L:lower, O:outer, I:inner, shape = [X[], Y[]]
    def __init__(self, shapeUO, shapeUI, shapeLO, shapeLI, N):
        self.lashU = eyelash_v2(shapeUO[0], shapeUO[1], shapeUI[0], shapeUI[1], N)
        self.lashL = eyelash_v2(shapeLO[0], shapeLO[1], shapeLI[0], shapeLI[1], N)
        self.shapeI = []
        for i in range(len(shapeUI[0])):
            self.shapeI.append([shapeUI[0][i], shapeUI[1][i]])
        for i in range(len(shapeLI[0]) - 2):
            self.shapeI.append([shapeLI[0][i + 1], shapeLI[1][i + 1]])
        self.shapeO = []
        for i in range(len(shapeUO[0])):
            self.shapeO.append([shapeUO[0][i], shapeUO[1][i]])
        for i in range(len(shapeLO[0])):
            self.shapeO.append([shapeLO[0][i], shapeLO[1][i]])
        self.setPolygon()

    def setShapeI(self, ind, X, Y): #内側の形 [[x1, y1], [x2, y2], ...]
        self.shapeI[ind] = [X, Y]
        if ind < len(self.lashU.shapeInnerX):
            self.lashU.setHandleInner(ind, X, Y)
            if ind == 0 or ind == len(self.lashU.shapeInnerX)-1:
                self.lashL.setHandleInner(ind, X, Y)
                print(self.lashL.shapeInnerY)
        else:
            self.lashL.setHandleInner(ind-len(self.lashU.shapeInnerX)+1, X, Y)

    def setShapeO(self, ind, X, Y): #外側の形 [[x1, y1], [x2, y2], ...]
        self.shapeO[ind] = [X, Y]
        if ind < len(self.lashU.shapeInnerX):
            self.lashU.setHandleOuter(ind, X, Y)
        else:
            self.lashL.setHandleOuter(ind-len(self.lashU.shapeInnerX), X, Y)

    def getPolygon(self):
        polygon = []
        lenU = len(self.lashU.shapeInnerXs)
        lenL = len(self.lashL.shapeInnerXs)
        for i in range(lenU):
            polygon.append([self.lashU.shapeInnerXs[i], self.lashU.shapeInnerYs[i]])
        for i in range(lenL - 2):
            polygon.append([self.lashL.shapeInnerXs[lenL-2-i], self.lashL.shapeInnerYs[lenL-2-i]])

        return polygon

    def setPolygon(self):
        self.polygon = self.getPolygon()

    def draw(self, canvas, color, tag):
        isInsideUouters = []
        isInsideLouters = []
        for i in range(len(self.lashU.shapeOuterXs)):
            isInsideUouters.append(isInside([self.lashU.shapeOuterXs[i],self.lashU.shapeOuterYs[i]], self.polygon))
        for i in range(len(self.lashL.shapeOuterXs)):
            isInsideLouters.append(isInside([self.lashL.shapeOuterXs[i],self.lashL.shapeOuterYs[i]], self.polygon))
        
        canvas.delete(tag)
        for i in range(self.lashU.length-1):
            if (not isInsideUouters[i]) and (not isInsideUouters[i+1]):
                self.lashU.draw(i, canvas, color, tag)

        for i in range(self.lashL.length-1):
            if (not isInsideLouters[i]) and (not isInsideLouters[i+1]):
                self.lashL.draw(i, canvas, color, tag)

    def drawWhite(self, canvas, color, tag):
        canvas.delete(tag)
        canvasW = 640
        canvasH = 480
        lenU = len(self.lashU.shapeInnerXs)
        for i in range(lenU-1):
            canvas.create_polygon(canvasW/(lenU-1)*i, 0,
            canvasW/(lenU-1)*(i+1), 0,
            self.lashU.shapeInnerXs[i+1], self.lashU.shapeInnerYs[i+1],
            self.lashU.shapeInnerXs[i], self.lashU.shapeInnerYs[i], 
            fill=color, tag = tag)

        lenL = len(self.lashL.shapeInnerXs)
        for i in range(lenU-1):
            canvas.create_polygon(canvasW/(lenU-1)*i, canvasH,
            canvasW/(lenU-1)*(i+1), canvasH,
            self.lashL.shapeInnerXs[i+1], self.lashL.shapeInnerYs[i+1],
            self.lashL.shapeInnerXs[i], self.lashL.shapeInnerYs[i], 
            fill=color, tag = tag)


class Application_EVGUI(tkinter.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title('eye_projecter')
        self.pack()
        self.create_widgets()

    def create_widgets(self):

        self.labelEyelash = tkinter.Label(self, text="eyelash")
        self.labelEyelash.grid(row=2, column=0)

        self.handleInd = tkinter.IntVar()
        self.handleInd.set(0)
        self.selhandleInd = tkinter.Spinbox(self, from_=0, to=100, increment=1, textvariable=self.handleInd)
        self.selhandleInd.grid(row=2, column=1)

        self.reloadButton = tkinter.Button(self, text='reload', command=self.reloadEyelash)
        self.reloadButton.grid(row=2, column=2)

        self.labelPpl = tkinter.Label(self, text="pupil")
        self.labelPpl.grid(row=3, column=0)

        self.handlePplInd = tkinter.IntVar()
        self.handlePplInd.set(0)
        self.selhandlePplInd = tkinter.Spinbox(self, from_=0, to=3, increment=1, textvariable=self.handlePplInd)
        self.selhandlePplInd.grid(row=3, column=1)

        self.updateButton = tkinter.Button(self, text='reload', command=self.reloadPpl)
        self.updateButton.grid(row=3, column=2)

        self.labelELLow = tkinter.Label(self, text="eyelashLow")
        self.labelELLow.grid(row=4, column=0)

        self.ELLowInd = tkinter.IntVar()
        self.ELLowInd.set(0)
        self.selELLowInd = tkinter.Spinbox(self, from_=0, to=4, increment=1, textvariable=self.ELLowInd)
        self.selELLowInd.grid(row=4, column=1)

        self.updateELLowButton = tkinter.Button(self, text='reload', command=self.reloadELLow)
        self.updateELLowButton.grid(row=4, column=2)

        self.tagOrder = ["refImg", "handleI", "handleO","pointPpl", "eyelash", "eyelid", "iris"]

        shapeUOx = [80, 200, 320, 440, 560]
        shapeUOy = [230, 110, 110, 110, 230]
        shapeUIx = [80, 200, 320, 440, 560]
        shapeUIy = [240, 120, 120, 120, 240]

        shapeLOx = [80, 200, 320, 440, 560]
        shapeLOy = [250, 370, 370, 370, 250]
        shapeLIx = [80, 200, 320, 440, 560]
        shapeLIy = [240, 360, 360, 360, 240]

        shapeUO = [shapeUOx, shapeUOy]
        shapeUI = [shapeUIx, shapeUIy]
        shapeLO = [shapeLOx, shapeLOy]
        shapeLI = [shapeLIx, shapeLIy]


        self.N = 50

        self.irisRad = 100
        self.handlePpl = [[50, 0], [100, 50], [50, 100], [0, 50]]

        self.mouseXY = [None, None]

        self.pointRad = 3
        self.pointVisible = True

        self.test_canvas = tkinter.Canvas(self, bg='white', width=640, height=480)
        self.test_canvas.grid(row=1, column=0, columnspan=4)

        self.test_canvas.bind('<Button-1>', self.mouseL)
        self.test_canvas.bind('<B1-Motion>', self.mouseL)
        self.test_canvas.bind('<ButtonRelease-1>', self.mouseLRel)

        self.test_canvas.bind('<Button-3>', self.mouseR)
        self.test_canvas.bind('<B3-Motion>', self.mouseR)

        self.test_canvas.bind('<Motion>', self.getMouseXY)

        self.master.bind('<KeyPress>', self.keyPress)

        self.eye_canvas = tkinter.Canvas(self, bg='white', width=101, height=101)
        self.eye_canvas.grid(row=5, column=0)

        self.colButton = tkinter.Button(self, text='color', command=self.chooseCol)
        self.colButton.grid(row=5, column=1)

        self.eyeBlackRad = tkinter.IntVar()
        self.eyeBlackRad.set(0)
        self.eyeBlackScale = tkinter.Scale(self, variable=self.eyeBlackRad, command=self.updateEyeBlack, orient='horizontal', from_=0, to=100)
        self.eyeBlackScale.grid(row=5, column=2)

        self.labelTarget = tkinter.Label(self, text="target")
        self.labelTarget.grid(row=0, column=0)

        self.targetInd = tkinter.IntVar()
        self.targetInd.set(0)
        self.seltargetInd = tkinter.Spinbox(self, from_=0, to=4, increment=1, textvariable=self.targetInd)
        self.seltargetInd.grid(row=0, column=1)

        self.saveButton = tkinter.Button(self, text='save', command=self.saveData)
        self.saveButton.grid(row=0, column=2)

        self.loadButton = tkinter.Button(self, text='load', command=self.loadData)
        self.loadButton.grid(row=0, column=3)

        self.refInd = tkinter.IntVar()
        self.refInd.set(0)
        self.selRefInd = tkinter.Spinbox(self, from_=1, to=143, increment=1, textvariable=self.refInd, command=self.dispRefImg)
        self.selRefInd.grid(row=6, column=0)
        self.dispRefImg()

        self.refVisBtn = tkinter.Button(self, text='visible', command=self.toggleRefVisible)
        self.refVisBtn.grid(row=6, column=1)

        self.refTransparency = tkinter.IntVar()
        self.refTransparency.set(0)
        self.refTraScale = tkinter.Scale(self, variable=self.refTransparency, command=self.updateTransparency, orient='horizontal', from_=0, to=255)
        self.refTraScale.grid(row=6, column=2)
        self.updateTransparency()

        #tkinter.Spinbox(self, from_=1, to=100, increment=1, textvariable=self.refInd, command=self.updateRefItr)
        self.refItr = tkinter.IntVar()
        self.refItr.set(108)
        self.refItrScale = tkinter.Spinbox(self, from_=108, to=143, increment=1, textvariable=self.refItr, command=self.updateRefItr)
        self.refItrScale.grid(row=6, column=3)

        self.outImgBtn = tkinter.Button(self, text='output', command=self.outputCurrentImg)
        self.outImgBtn.grid(row=5, column=3)

        self.refVisible = True

        self.eyeColorBtn = tkinter.Button(self, text='eyeColor', command=self.applyEyeColor)
        self.eyeColorBtn.grid(row=7, column=0)


        #self.dispRefImg()
        ref = Image.open('output/outputEyeImg.png')
        ref.putalpha(255)
        ref = ref.resize((640, 480))
        ref.save('temp_img/temp_ref.png')
        self.touka = tkinter.PhotoImage(file='temp_img/temp_ref.png')
        self.test_canvas.create_image(2,2,image=self.touka,anchor=tkinter.NW,tag="refImg")

        self.eye1 = eye(shapeUO, shapeUI, shapeLO, shapeLI, self.N)
        self.polygon = self.eye1.getPolygon()

        self.eye1.draw(self.test_canvas, "black", "eyelash")
        self.eye1.drawWhite(self.test_canvas, "linen", "eyelid")
        self.reloadPpl()
        for point in self.eye1.shapeI:
            createOvalEZ(self.test_canvas, point[0], point[1], self.pointRad, "red", "handleI")
        for point in self.eye1.shapeO:
            createOvalEZ(self.test_canvas, point[0], point[1], self.pointRad, "blue", "handleO")

        self.loadURL('json_data/avg_v2.json')
        #self.loadURL('json_data/vec_eigSpace.json')
        
        sortDrawOrder(self.test_canvas, self.tagOrder)


    def mouseL(self, event):

        if self.targetInd.get() == 0:

            self.eye1.setShapeI(self.handleInd.get(), event.x, event.y)
            self.eye1.draw(self.test_canvas, "black", "eyelash")
            self.eye1.drawWhite(self.test_canvas, "linen", "eyelid")

            self.test_canvas.delete("handleI")
            for point in self.eye1.shapeI:
                createOvalEZ(self.test_canvas, point[0], point[1], self.pointRad, "red", "handleI")
            
            self.eye1.setPolygon()

            sortDrawOrder(self.test_canvas, self.tagOrder)


        elif self.targetInd.get() == 1:

            self.handlePpl[self.handlePplInd.get()] = [event.x, event.y]

            sortDrawOrder(self.test_canvas, self.tagOrder)

        elif self.targetInd.get() == 2:

            self.eye1.setShapeO(self.ELLowInd.get(), event.x, event.y)
            self.eye1.draw(self.test_canvas, "black", "eyelash")

            self.test_canvas.delete("handleO")
            for point in self.eye1.shapeO:
                createOvalEZ(self.test_canvas, point[0], point[1], self.pointRad, "blue", "handleO")

            sortDrawOrder(self.test_canvas, self.tagOrder)

    def mouseLRel(self, event):
        if self.targetInd.get() == 1:
            self.reloadPpl()
            #print('a')

        sortDrawOrder(self.test_canvas, self.tagOrder)

    def mouseR(self, event):

        if self.targetInd.get() == 0:
            self.tX2[self.handleInd.get()] = event.x
            self.tY2[self.handleInd.get()] = event.y
            self.test.setShapeInner(self.tX2, self.tY2)
            self.test_canvas.delete("tXY2point")
            self.test.draw(self.test_canvas, color="gray", tag="test")
            for i in range(5):
                createOvalEZ(self.test_canvas, self.tX2[i], self.tY2[i], self.pointRad, "blue", "tXY2point")



        if self.targetInd.get() == 2:
            if event.y > self.ELLowY[self.ELLowInd.get()]:
                self.ELLowY2[self.ELLowInd.get()] = event.y
            else:
                self.ELLowY2[self.ELLowInd.get()] = self.ELLowY[self.ELLowInd.get()]

            self.ELLowRad[self.ELLowInd.get()] = self.ELLowY2[self.ELLowInd.get()] - self.ELLowY[self.ELLowInd.get()]

            print(self.ELLowRad)

            self.reloadELLow()

    def reloadEyelash(self):

        # まつ毛の形状更新
        handleY = (np.array(self.handleY) + np.array(self.handleY2))/2
        handleY = handleY.astype(np.int64)
        self.eyelash.setShape(self.handleX, handleY)

        # まつ毛の太さ更新
        rad = (np.array(self.handleY2) - np.array(self.handleY))/2
        rad = rad.astype(np.int64)
        #print(rad)
        self.eyelash.setRad(rad)

        # まつ毛, まぶた描画
        self.test_canvas.delete("lids")
        self.test_canvas.delete("point")
        for i in range(self.N):

            posX = self.eyelash.shapeXs[i]
            posY = self.eyelash.shapeYs[i]

            rad = self.eyelash.rads[i]

            if i != 0 and i%20 == 0:
                j = i / 20
                self.test_canvas.create_polygon((j-1)*64, 0, j*64, 0, posX, posY, self.eyelash.shapeXs[i-20], self.eyelash.shapeYs[i-20], fill="linen", tag = "lids")

            self.test_canvas.create_polygon(0,0,0,480,self.eyelash.shapeXs[0], self.eyelash.shapeYs[0], fill="linen", tag = "lids")
            self.test_canvas.create_polygon(640,0,640,480,self.eyelash.shapeXs[self.N-1], self.eyelash.shapeYs[self.N-1], fill="linen", tag = "lids")
            
            if rad > 0:
                self.test_canvas.create_oval(posX - rad, posY - rad, posX + rad, posY + rad, fill="black", tag = "point", width=0)
        
        self.test_canvas.delete("pointEL")
        for i in range(5):
            self.test_canvas.create_oval(self.handleX[i]-self.pointRad, self.handleY2[i]-self.pointRad, self.handleX[i]+self.pointRad, self.handleY2[i]+self.pointRad, fill="blue", tag = "pointEL", width=0)
            self.test_canvas.create_oval(self.handleX[i]-self.pointRad, self.handleY[i]-self.pointRad, self.handleX[i]+self.pointRad, self.handleY[i]+self.pointRad, fill="red", tag = "pointEL", width=0)
        
        self.test_canvas.tag_raise("lids")
        self.test_canvas.tag_raise("lidsLow")
        self.test_canvas.tag_raise("point")
        self.test_canvas.tag_raise("pointLow")
        self.test_canvas.tag_raise("pointEL")
        self.test_canvas.tag_raise("pointELLow")
        self.test_canvas.tag_raise("refImg")
    
    def reloadELLow(self):

        # まつ毛の形状更新
        handleY = (np.array(self.ELLowY) + np.array(self.ELLowY2))/2
        handleY = handleY.astype(np.int64)
        self.eyelashLow.setShape(self.ELLowX, handleY)

        # まつ毛の太さ更新
        rad = (np.array(self.ELLowY2) - np.array(self.ELLowY))/2
        rad = rad.astype(np.int64)
        #print(rad)
        self.eyelashLow.setRad(rad)

        # まつ毛描画
        self.test_canvas.delete("pointLow")
        self.test_canvas.delete("lidsLow")
        for i in range(self.N):

            posX = self.eyelashLow.shapeXs[i]
            posY = self.eyelashLow.shapeYs[i]

            rad = self.eyelashLow.rads[i]

            if rad > 0:
                self.test_canvas.create_oval(posX - rad, posY - rad, posX + rad, posY + rad, fill="black", tag = "pointLow", width=0)

            if i != 0 and i%20 == 0:
                j = i / 20
                self.test_canvas.create_polygon((j-1)*64, 480, j*64, 480, posX, posY, self.eyelashLow.shapeXs[i-20], self.eyelashLow.shapeYs[i-20], fill="linen", tag = "lidsLow")
        
        self.test_canvas.delete("pointELLow")
        for i in range(5):
            self.test_canvas.create_oval(self.ELLowX[i]-self.pointRad, self.ELLowY2[i]-self.pointRad, self.ELLowX[i]+self.pointRad, self.ELLowY2[i]+self.pointRad, fill="blue", tag = "pointELLow", width=0)
            self.test_canvas.create_oval(self.ELLowX[i]-self.pointRad, self.ELLowY[i]-self.pointRad, self.ELLowX[i]+self.pointRad, self.ELLowY[i]+self.pointRad, fill="red", tag = "pointELLow", width=0)

        self.test_canvas.tag_raise("lids")
        self.test_canvas.tag_raise("lidsLow")
        self.test_canvas.tag_raise("point")
        self.test_canvas.tag_raise("pointLow")
        self.test_canvas.tag_raise("pointEL")
        self.test_canvas.tag_raise("pointELLow")
        self.test_canvas.tag_raise("refImg")

    def chooseCol(self):
        col = colorchooser.askcolor()
        self.colButton.config(bg=col[1])
        self.eyeCol_1 = np.array([col[0][2], col[0][1], col[0][0]])

        self.iris_tex = circleTex(self.irisRad, np.array([0,0,0]), self.eyeCol_1, np.array([-1,-1,-1])*70, np.array([1,1,1])*120)
        cv2.imwrite('temp_img/iris_tex.png', self.iris_tex)

        self.irisTex = tkinter.PhotoImage(file='temp_img/iris_tex.png')
        self.eye_canvas.create_image(2,2,image=self.irisTex,anchor=tkinter.NW)

    def reloadPpl(self):
        self.test_canvas.delete("iris")

        tex = cv2.imread('temp_img/iris_tex.png')
        irisMesh = DmeshLib.mesh(4, 4, 2*self.irisRad+1, 2*self.irisRad+1, tex)
        irisMesh.setHandlesOrg(np.array([[[self.irisRad, 0]], [[2*self.irisRad+1, self.irisRad]], [[self.irisRad, 2*self.irisRad+1]], [[0, self.irisRad]]]))

        irisMesh.setHandlesDfm(np.array([[self.handlePpl[0]], [self.handlePpl[1]], [self.handlePpl[2]], [self.handlePpl[3]]]))
        #irisMesh.setHandlesDfm(np.array([[[320, 0]], [[640, 240]], [[320, 480]], [[0, 240]]]))
        irisMesh.applyHandles()
        dfmTex = irisMesh.deform(w=640, h=480)
        cv2.imwrite('temp_img/iris_tex_dfm.png', dfmTex)

        self.irisTexDfm = tkinter.PhotoImage(file='temp_img/iris_tex_dfm.png')
        self.test_canvas.create_image(2,2,image=self.irisTexDfm,anchor=tkinter.NW,tag="iris")
        self.test_canvas.tag_lower("iris")

        self.test_canvas.delete("pointPpl")
        for point in self.handlePpl:
            createOvalEZ(self.test_canvas, point[0], point[1], self.pointRad, "green", "pointPpl")

    def getMouseXY(self, event):
        self.mouseXY = [event.x, event.y]
        #print(isInside(self.mouseXY, self.eye1.polygon))

    def applyEyeColor(self):
        dfmTex = cv2.imread('temp_img/iris_tex_dfm.png')
        dfmTex = cv2.resize(dfmTex, (64,48))
        i = self.refInd.get()
        orgTex = cv2.imread('data_eyes_test/'+str(i).zfill(3)+'.png')
        #orgTex = cv2.resize(orgTex, (640, 480))
        eyeRef = cv2.imread('output/_wv_output.png')
        #eyeRef = cv2.imread('data_eyes_test/'+str(self.refItr.get()).zfill(3)+'.png')
        #eyeRef = cv2.imread('data_eyes_test/'+str(i).zfill(3)+'.png')
        
        colorEyeTex = np.copy(dfmTex)

        eyeCol1 = np.array([0,0,0])
        eyeCol2 = np.array([0,0,0])
        eyeCol3 = np.array([0,0,0])
        numFilt1 = 0
        numFilt2 = 0
        numFilt3 = 0

        for i in range(48):
            for j in range(64):
                if dfmTex[i][j][0] == 255 and dfmTex[i][j][1] == 0:
                    numFilt1 += 1
                    eyeCol1 = eyeCol1 + eyeRef[i][j]
                if dfmTex[i][j][1] == 255 and dfmTex[i][j][2] == 0:
                    numFilt2 += 1
                    eyeCol2 = eyeCol2 + eyeRef[i][j]
                if dfmTex[i][j][2] == 255 and dfmTex[i][j][0] == 0:
                    numFilt3 += 1
                    eyeCol3 = eyeCol3 + eyeRef[i][j]
                if np.all(dfmTex[i][j] == 255):
                    orgTex[i][j] = np.array([255, 255, 255])

        eyeCol1 = eyeCol1 / numFilt1
        eyeCol1 = eyeCol1.astype(np.uint8)
        eyeCol2 = eyeCol2 / numFilt2
        eyeCol2 = eyeCol2.astype(np.uint8)
        eyeCol3 = eyeCol3 / numFilt3
        eyeCol3 = eyeCol3.astype(np.uint8)

        newIrisTex = circleTex_2(self.irisRad, self.eyeBlackRad.get(), np.array([128,128,128]), np.array([16,16,16]), eyeCol1, eyeCol2, eyeCol3, True)

        irisMesh = DmeshLib.mesh(4, 4, 2*self.irisRad+1, 2*self.irisRad+1, newIrisTex)
        irisMesh.setHandlesOrg(np.array([[[self.irisRad, 0]], [[2*self.irisRad+1, self.irisRad]], [[self.irisRad, 2*self.irisRad+1]], [[0, self.irisRad]]]))

        irisMesh.setHandlesDfm(np.array([[self.handlePpl[0]], [self.handlePpl[1]], [self.handlePpl[2]], [self.handlePpl[3]]]))
        #irisMesh.setHandlesDfm(np.array([[[320, 0]], [[640, 240]], [[320, 480]], [[0, 240]]]))
        irisMesh.applyHandles()
        dfmTex = irisMesh.deform(w=640, h=480)

        cv2.imwrite('temp_img/iris_tex_only.png', orgTex)
        cv2.imwrite('temp_img/iris_tex_dfm.png', dfmTex)

        self.irisTexDfm = tkinter.PhotoImage(file='temp_img/iris_tex_dfm.png')
        self.test_canvas.create_image(2,2,image=self.irisTexDfm,anchor=tkinter.NW,tag="iris")
        self.test_canvas.tag_lower("iris")

        self.test_canvas.delete("pointPpl")
        for point in self.handlePpl:
            createOvalEZ(self.test_canvas, point[0], point[1], self.pointRad, "green", "pointPpl")

    def keyPress(self, event):
        print(event.keycode, self.mouseXY)

        if event.keycode == 65:
            if self.targetInd.get() == 0:
                minInd = getNearest(self.mouseXY, self.eye1.shapeI)
                self.handleInd.set(minInd)

            elif self.targetInd.get() == 2:
                minInd = getNearest(self.mouseXY, self.eye1.shapeO)
                self.ELLowInd.set(minInd)

            elif self.targetInd.get() == 1:
                minInd = getNearest(self.mouseXY, self.handlePpl)
                self.handlePplInd.set(minInd)
                #print('a')
            
        elif event.keycode == 86:
            if self.pointVisible:
                self.test_canvas.delete("pointPpl")
                self.test_canvas.delete("pointEL")
                self.test_canvas.delete("pointELLow")
                self.pointVisible = False
            else:
                for i in range(5):
                    self.test_canvas.create_oval(self.handleX[i]-self.pointRad, self.handleY2[i]-self.pointRad, self.handleX[i]+self.pointRad, self.handleY2[i]+self.pointRad, fill="blue", tag = "pointEL", width=0)
                    self.test_canvas.create_oval(self.handleX[i]-self.pointRad, self.handleY[i]-self.pointRad, self.handleX[i]+self.pointRad, self.handleY[i]+self.pointRad, fill="red", tag = "pointEL", width=0)
                for i in range(5):
                    self.test_canvas.create_oval(self.ELLowX[i]-self.pointRad, self.ELLowY2[i]-self.pointRad, self.ELLowX[i]+self.pointRad, self.ELLowY2[i]+self.pointRad, fill="blue", tag = "pointELLow", width=0)
                    self.test_canvas.create_oval(self.ELLowX[i]-self.pointRad, self.ELLowY[i]-self.pointRad, self.ELLowX[i]+self.pointRad, self.ELLowY[i]+self.pointRad, fill="red", tag = "pointELLow", width=0)
                for i in range(4):
                    self.test_canvas.create_oval(self.handlePpl[i][0]-self.pointRad, self.handlePpl[i][1]-self.pointRad, self.handlePpl[i][0]+self.pointRad, self.handlePpl[i][1]+self.pointRad, fill="green", tag = "pointPpl", width=0)
                self.pointVisible = True
            #print(minInd)

        elif event.keycode == 37:
            self.test_canvas.move("refImg", -1, 0)
        elif event.keycode == 38:
            self.test_canvas.move("refImg", 0, -1)
        elif event.keycode == 39:
            self.test_canvas.move("refImg", 1, 0)
        elif event.keycode == 40:
            self.test_canvas.move("refImg", 0, 1)

    def dispRefImg(self):
        self.test_canvas.delete("refImg")
        i = self.refInd.get()
        ref = Image.open('data_eyes_test/' + str(i).zfill(3) + '.png')
        ref.putalpha(128)
        ref = ref.resize((640, 480))
        ref.save('temp_img/temp_ref.png')

        self.touka = tkinter.PhotoImage(file='temp_img/temp_ref.png')
        self.test_canvas.create_image(2,2,image=self.touka,anchor=tkinter.NW,tag="refImg")

    def updateTransparency(self, event=None):
        self.test_canvas.delete("refImg")
        ref = Image.open('temp_img/temp_ref.png')
        ref.putalpha(self.refTransparency.get())
        ref = ref.resize((640, 480))
        ref.save('temp_img/temp_ref.png')

        self.touka = tkinter.PhotoImage(file='temp_img/temp_ref.png')
        self.test_canvas.create_image(2,2,image=self.touka,anchor=tkinter.NW,tag="refImg")

    def updateEyeBlack(self, event):
        self.iris_tex = circleTex_2(self.irisRad, self.eyeBlackRad.get(), np.array([128,128,128]), np.array([16,16,16]))
        cv2.imwrite('temp_img/iris_tex.png', self.iris_tex)

        self.irisTex = tkinter.PhotoImage(file='temp_img/iris_tex.png')
        self.eye_canvas.create_image(2,2,image=self.irisTex,anchor=tkinter.NW)

        self.reloadPpl()

    def updateRefItr(self):
        if self.refItr.get() == 108:
            self.loadURL('json_data/avg_v2.json')
        elif self.refItr.get() == 109:
            self.loadURL('json_data/vec_eigSpace.json')
        else:
            itr = str(self.refItr.get())
            self.loadURL('test_result/json/test'+str(itr)+'.json')
            print('test_result/json/test'+str(itr)+'.json')

            i = int(self.refItr.get())

            if(i != 0):
                ref = Image.open('data_eyes_test/' + str(itr).zfill(3) + '.png')
                ref.putalpha(128)
                ref = ref.resize((640, 480))
                ref.save('temp_img/temp_ref.png')
                self.touka = tkinter.PhotoImage(file='temp_img/temp_ref.png')
                self.test_canvas.create_image(2,2,image=self.touka,anchor=tkinter.NW,tag="refImg")

    def toggleRefVisible(self):
        if self.refVisible:
            self.test_canvas.delete("refImg")
            self.refVisible = False
        else:
            self.test_canvas.create_image(2,2,image=self.touka,anchor=tkinter.NW,tag="refImg")
            self.refVisible = True

    def saveData(self):
        data = {}
        data['shapeUIx'] = self.eye1.lashU.shapeInnerX
        data['shapeUIy'] = self.eye1.lashU.shapeInnerY
        data['shapeUOx'] = self.eye1.lashU.shapeOuterX
        data['shapeUOy'] = self.eye1.lashU.shapeOuterY
        data['shapeLIx'] = self.eye1.lashL.shapeInnerX
        data['shapeLIy'] = self.eye1.lashL.shapeInnerY
        data['shapeLOx'] = self.eye1.lashL.shapeOuterX
        data['shapeLOy'] = self.eye1.lashL.shapeOuterY
        data['pplXY'] = self.handlePpl

        i = self.refInd.get()
        with open('json_data/'+str(i).zfill(3)+'_v2.json', 'w') as f:
            json.dump(data, f)
        print("saved.")

    def loadData(self):
        data = None
        i = self.refInd.get()
        with open('json_data/'+str(i).zfill(3)+'_v2.json') as f:
            data = json.load(f)

        shapeUO = [data['shapeUOx'], data['shapeUOy']]
        shapeUI = [data['shapeUIx'], data['shapeUIy']]
        shapeLO = [data['shapeLOx'], data['shapeLOy']]
        shapeLI = [data['shapeLIx'], data['shapeLIy']]
        self.eye1 = eye(shapeUO, shapeUI, shapeLO, shapeLI, self.N)
        self.handlePpl = data['pplXY']

        self.test_canvas.delete("handleI")
        self.test_canvas.delete("handleO")
        self.test_canvas.delete("pointPpl")
        for point in self.eye1.shapeI:
            createOvalEZ(self.test_canvas, point[0], point[1], self.pointRad, "red", "handleI")
        for point in self.eye1.shapeO:
            createOvalEZ(self.test_canvas, point[0], point[1], self.pointRad, "blue", "handleO")
        for point in self.handlePpl:
            createOvalEZ(self.test_canvas, point[0], point[1], self.pointRad, "green", "pointPpl")
        
        self.eye1.draw(self.test_canvas, "black", "eyelash")
        self.eye1.drawWhite(self.test_canvas, "linen", "eyelid")
        self.reloadPpl()
        sortDrawOrder(self.test_canvas, self.tagOrder)

    def loadURL(self, url, polygon = None):
        data = None
        i = self.refInd.get()
        with open(url) as f:
            data = json.load(f)

        shapeUO = [data['shapeUOx'], data['shapeUOy']]
        shapeUI = [data['shapeUIx'], data['shapeUIy']]
        shapeLO = [data['shapeLOx'], data['shapeLOy']]
        shapeLI = [data['shapeLIx'], data['shapeLIy']]
        LOs = [0, 1]
        if True:
            for l in LOs: # 目頭側の下まつげを強制的に内側に
                shapeLO[0][l] = shapeLI[0][l]+5
                shapeLO[1][l] = shapeLI[1][l]-5
        self.eye1 = eye(shapeUO, shapeUI, shapeLO, shapeLI, self.N)
        self.handlePpl = data['pplXY']

        self.test_canvas.delete("handleI")
        self.test_canvas.delete("handleO")
        self.test_canvas.delete("pointPpl")
        for point in self.eye1.shapeI:
            createOvalEZ(self.test_canvas, point[0], point[1], self.pointRad, "red", "handleI")
        for point in self.eye1.shapeO:
            createOvalEZ(self.test_canvas, point[0], point[1], self.pointRad, "blue", "handleO")
        for point in self.handlePpl:
            createOvalEZ(self.test_canvas, point[0], point[1], self.pointRad, "green", "pointPpl")
        
        self.eye1.draw(self.test_canvas, "black", "eyelash")
        self.eye1.drawWhite(self.test_canvas, "linen", "eyelid")

        self.reloadPpl()
        sortDrawOrder(self.test_canvas, self.tagOrder)

    def outputCurrentImg(self):
        self.test_canvas.postscript(file='temp_img/out.ps', colormode='color')
        psimage=Image.open('temp_img/out.ps')
        psimage.save('temp_img/vectorImg.png')
        





root = tkinter.Tk()
app = Application_EVGUI(master=root)
app.mainloop()