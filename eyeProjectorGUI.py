# 入力スケッチを受け付け、データから新しい目データを生成するGUI

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
from PIL import ImageTk, Image

from projectInput_vector import project_withVector
import DmeshLib as DMesh
from utlLib import transferColor, createOvalEZ, blendPreview, calcHandleDiff
import preData as pre
import preData2 as pre2
from preData3 import project_autoHandleGen_EM, findEdge

def colorCord(R,G,B): #整数値RGBをカラーコードに
    return('#'+ format(R, '02x')+ format(G, '02x')+ format(B, '02x'))

def EMPCAHandGenFromTestNumber(testNumber):
    truehand = pre.handlesArr[testNumber-1]
    handles = []
    handles.append(truehand[1])
    handles.append(truehand[2])
    handles.append(truehand[3])
    handles.append(truehand[12])
    handles = np.array(handles)
    pngImg = cv2.imread('data_eyes_test/'+str(testNumber).zfill(3)+'.png')
    print(pngImg)
    pngImgGray = cv2.cvtColor(pngImg, cv2.COLOR_BGR2GRAY)
    newImgVec, newHandleVec = project_autoHandleGen_EM(pngImgGray,handles)
    #print("handleTest",newHandleVec, handles)
    newHandleVec = newHandleVec.reshape(13,1,2)
    return pngImg, newHandleVec


class Application(tkinter.Frame): #GUI
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title('eye_projecter')
        self.pack()
        self.create_widgets()
        self.setup()
        self.handles = []
        self.handleNum = 0
        self.generateNum = 0

    def create_widgets(self): #ウィジェットを並べる
        self.vr = tkinter.IntVar()
        self.vr.set(1)
        self.write_radio = tkinter.Radiobutton(self, text='write', variable=self.vr, value=1, command=self.change_radio)
        self.write_radio.grid(row=0, column=0)
        self.erase_radio = tkinter.Radiobutton(self, text='erase', variable=self.vr, value=2, command=self.change_radio)
        self.erase_radio.grid(row=0, column=1)

        self.clear_button = tkinter.Button(self, text='clear all', command=self.clear_canvas)
        self.clear_button.grid(row=0, column=2)

        self.save_button = tkinter.Button(self, text='generate', command=self.save_canvas)
        self.save_button.grid(row=0, column=3)

        self.test_button = tkinter.Button(self, text='test', command=self.genTest)
        self.test_button.grid(row=0, column=4)

        self.thickness = tkinter.IntVar()
        self.thickness.set(35)
        self.slider = tkinter.Scale(self, label='thickness', from_=10, to=50, orient=tkinter.HORIZONTAL, variable=self.thickness)
        self.slider.grid(row=1, column=0)

        self.colR = tkinter.IntVar()
        self.colR.set(0)
        self.selR = tkinter.Scale(self, label='R', from_=0, to=255, variable=self.colR)
        self.selR.grid(row=1, column=1)

        self.colG = tkinter.IntVar()
        self.colG.set(0)
        self.selG = tkinter.Scale(self, label='G', from_=0, to=255, variable=self.colG)
        self.selG.grid(row=1, column=2)

        self.colB = tkinter.IntVar()
        self.colB.set(0)
        self.selB = tkinter.Scale(self, label='B', from_=0, to=255, variable=self.colB)
        self.selB.grid(row=1, column=3)

        self.mode = tkinter.IntVar()
        self.mode.set(1)
        self.modeSelect = tkinter.Scale(self, label='mode', from_=0, to=1, orient=tkinter.HORIZONTAL, variable=self.mode)
        self.modeSelect.grid(row=2, column=0)

        self.textName = tkinter.Entry(self, width=20)
        self.textName.grid(row=2, column=1)

        self.handlePosCanvas = tkinter.Canvas(self, bg='white', width=64, height=48)
        #self.handlePosCanvas.grid(row=2, column=2)

        #self.autoHandleBtn = tkinter.Button(self, text='generate handle', command=self.generateHandle)
        #self.autoHandleBtn.grid(row=2, column=3)

        self.autoHandleBtn = tkinter.Button(self, text='generate handle (EMPCA)', command=self.generateHandle_EM)
        #self.autoHandleBtn.grid(row=2, column=4)

        self.test_canvas = tkinter.Canvas(self, bg='white', width=640, height=480)
        self.test_canvas.grid(row=3, column=0, rowspan=3, columnspan=4)
        self.test_canvas.bind('<B1-Motion>', self.paint)
        self.test_canvas.bind('<Button-1>', self.register_RDL)
        self.test_canvas.bind('<ButtonRelease-1>', self.reset)
        self.test_canvas.bind('<Button-3>', self.setEyeCenterPos)

        self.testInd = tkinter.IntVar()
        self.testInd.set(1)
        self.testIndSelect = tkinter.Spinbox(self, from_=1, to=143, increment=1, textvariable=self.testInd, command=self.updateTestImg)
        self.testIndSelect.grid(row=2, column=2)
        self.updateTestImg()

        self.contRate = tkinter.DoubleVar()
        self.contRate.set(0.95)
        self.contRateSelect = tkinter.Scale(self, label='contRate', from_=0.3, to=0.99, resolution=0.01, orient=tkinter.HORIZONTAL, variable=self.contRate)
        self.contRateSelect.grid(row=2, column=3)

        self.refCanvas2 = tkinter.Canvas(self, bg='white', width=64, height=48)
        self.refCanvas2.grid(row=3, column=4)

        self.refInd2 = tkinter.IntVar()
        self.refInd2.set(1)
        self.refIndSelect2 = tkinter.Spinbox(self, from_=1, to=143, increment=1, textvariable=self.refInd2, command=self.updateRefImg2)
        self.refIndSelect2.grid(row=4, column=4)
        self.updateRefImg2()

        self.contRate2 = tkinter.DoubleVar()
        self.contRate2.set(0.3)
        self.contRateSelect2 = tkinter.Scale(self, label='contRate', from_=0.3, to=0.99, resolution=0.01, orient=tkinter.HORIZONTAL, variable=self.contRate2)
        self.contRateSelect2.grid(row=5, column=4)

        self.previewCanvas = tkinter.Canvas(self, bg='white', width=300, height=300)
        self.previewCanvas.grid(row=3, column=5, rowspan=3, columnspan=3)
        self.previewCanvas.bind('<B1-Motion>', self.previewB1Motion)

        self.previewInd = tkinter.IntVar()
        self.previewInd.set(1)
        self.previewIndSelect = tkinter.Spinbox(self, from_=1, to=143, increment=1, textvariable=self.previewInd, command=self.updatePreview)
        self.previewIndSelect.grid(row=6, column=5)

        self.previewEyeR = tkinter.BooleanVar()
        self.previewEyeR.set(False)
        self.previewToggle = tkinter.Checkbutton(self, text = "displayEyeR", command = self.updatePreview, variable = self.previewEyeR)
        #self.previewToggle.grid(row=4,column=5)

        self.previewEyeLCenter = (185, 144)
        self.previewEyeRCenter = (115, 144)

        self.previewEyeScale = tkinter.IntVar()
        self.previewEyeScale.set(100)
        self.previewEyeScaleSel = tkinter.Scale(self, label='eyeScale', from_=50, to=200, orient=tkinter.HORIZONTAL, variable=self.previewEyeScale, command=self.updatePreview)
        self.previewEyeScaleSel.grid(row=6, column=6)

        self.refCanvas = tkinter.Canvas(self, bg='white', width = 64, height=48)
        self.refCanvas.grid(row=6, column=0)

        self.refInd = tkinter.IntVar()
        self.refInd.set(1)
        self.selRefInd = tkinter.Spinbox(self, from_=1, to=143, increment=1, textvariable=self.refInd, command=self.updateRefImg)
        self.selRefInd.grid(row=6, column=1)
        ind = self.refInd.get()
        url = 'data_eyes/'+ str(ind).zfill(3)+ '.png'
        img = Image.open(url)
        self.refImg = ImageTk.PhotoImage(img)
        self.refCanvas.create_image(32,24,image=self.refImg)

        self.detailButton = tkinter.Button(self, text='add details', command=self.addDetails)
        self.detailButton.grid(row=6, column=2)

        self.outputCanvas = tkinter.Canvas(self, bg='white', width=128, height=96)
        self.outputCanvas.grid(row=7, column=0)

        self.outputCanvas2 = tkinter.Canvas(self, bg='white', width=128, height=96)
        self.outputCanvas2.grid(row=7, column=1)

        self.outputCanvas3 = tkinter.Canvas(self, bg='white', width=128, height=96)
        self.outputCanvas3.grid(row=7, column=2)

        self.outputCanvas4 = tkinter.Canvas(self, bg='white', width=128, height=96)
        self.outputCanvas4.grid(row=7, column=3)

        self.EVGUI = None

        self.handlePosCanvas.delete(tkinter.ALL)
        url = 'handlePos/handlePos1.png'
        img = Image.open(url)
        img = img.resize((64,48))
        self.handlePosImg = ImageTk.PhotoImage(img)
        self.handlePosCanvas.create_image(32,24,image=self.handlePosImg)

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.color = 'black'
        self.eraser_on = False
        #self.im = Image.new('RGB', (640, 480), 'white')
        #self.draw = ImageDraw.Draw(self.im)

    def change_radio(self):
        if self.vr.get() == 1:
            self.eraser_on = False
        else:
            self.eraser_on = True

    def clear_canvas(self): # 登録したハンドル, 描いた目削除
        self.test_canvas.delete(tkinter.ALL)
        self.handles = []
        self.handleNum = 0
        self.generateNum = 0

    def save_canvas(self): #入力を投影して出力を画像で保存 必ずgenerateHandle_EMを実行してから
        if self.mode.get() == 1:
            center = pre.handlesArr[self.testInd.get()-1][12][0]
            self.eyeCenterPos = [center[0]*10, center[1]*10]
            createOvalEZ(self.test_canvas, center[0]*10, center[1]*10, 5, "green", "eyeCenter")
        
        self.generateHandle_EM()

        refImg, refHandle = EMPCAHandGenFromTestNumber(self.refInd2.get())
        self.refs = [self.contRate2.get(), refImg, refHandle]

        self.test_canvas.postscript(file='temp_img/out.ps', colormode='color')
        psimage=Image.open('temp_img/out.ps')
        psimage.save('temp_img/sketch+anchor.png')
        self.test_canvas.delete("handleMark")
        self.test_canvas.delete("eyeCenter")
        if self.handleNum == 13:
            self.test_canvas.postscript(file='temp_img/out.ps', colormode='color')
            psimage=Image.open('temp_img/out.ps')

            print(psimage.size)
            psimage.save('temp_img/out.png')

            pngImg = cv2.imread('temp_img/out.png')
            pngImg = pngImg[2:362, 2:482]
            pngImg = cv2.resize(pngImg, (640,480))
            print(pngImg.shape)

            handles = np.array(self.handles)

            dx = 325 - handles[12][0][0]
            dy = 245 - handles[12][0][1]

            handles = handles + [[dx, dy]]
            handles = handles / 10

            afin_matrix = np.float32([[1,0,dx],[0,1,dy]])
            cv2.imwrite('output/idoumae.png', pngImg)
            pngImg = cv2.warpAffine(pngImg, afin_matrix, (640,480), borderValue = (230,240,250))
            cv2.imwrite('output/idougo.png', pngImg)
            print("dx dy: ",dx, dy)
            pngImg = cv2.resize(pngImg, (64,48))
            
            #pngImg = cv2.imread('temp_img/normalizedSketch')
            #pngImg = cv2.resize(pngImg, (64,48))
            
            #newEye = project(pngImg, handles)
            conts = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
            conts = []
            if self.mode.get() == 1:
                newEye_wv, newHandles_wv, newEyeMask_wv , self.eyeErrors= project_withVector(pngImg, handles,testNumber=self.testInd.get()-1,outputErrors=True,contRate=self.contRate.get(), isSelectEig=True)
                for cont in conts:
                    eye, handle, mask, _err = project_withVector(pngImg, handles,contRate=cont)
                    cv2.imwrite('output/contRateTest/'+self.textName.get()+str(cont)+'.png', eye)
            else:
                newEye_allEig, __newHandles_wv, __newEyeMask_wv , _err= project_withVector(pngImg, handles,contRate=self.contRate.get(),refs=self.refs,isSelectEig=False)
                newEye_wv, newHandles_wv, newEyeMask_wv , _err= project_withVector(pngImg, handles,contRate=self.contRate.get(),refs=self.refs,isSelectEig=True)
                for cont in conts:
                    eye, handle, mask, _err = project_withVector(pngImg, handles,contRate=cont,refs=self.refs)
                    cv2.imwrite('output/contRateTest/'+self.textName.get()+str(cont)+'.png', eye)
            self.outputHandles = newHandles_wv

            #cv2.imwrite('output/'+self.textName.get()+'_output.png', newEye)
            cv2.imwrite('output/'+self.textName.get()+'_input.png', pngImg)
            cv2.imwrite('output/'+self.textName.get()+'_wv_output.png', newEye_wv)
            cv2.imwrite('output/outputEyeImg.png', newEye_wv)
            cv2.imwrite('output/outputEyeImg_allEig.png', newEye_allEig)
            cv2.imwrite('output/projectedSketch.png', newEye_wv)
            cv2.imwrite('temp_img/eyeMask.png', newEyeMask_wv)

            # テスト用正解データ（アンカーポイント）
            testHandles = pre.handlesArr[self.testInd.get()-1]

            #newEye_TC = transferColor('output/'+self.textName.get()+'_wv_output.png', newHandles_wv, 'data_eyes/'+str(self.refInd.get()).zfill(3)+'.png', pre.handlesArr[self.refInd.get()-1])
            #cv2.imwrite('output/test_output.png', newEye_TC)

            self.outputCanvas.delete(tkinter.ALL)
            url = 'output/'+self.textName.get()+'_wv_output.png'
            img = Image.open(url)
            img = img.resize((128,96))
            self.outImg = ImageTk.PhotoImage(img)
            self.outputCanvas.create_image(64,48,image=self.outImg)

            self.outputCanvas4.delete(tkinter.ALL)
            url = 'output/outputEyeImg_allEig.png'
            img = Image.open(url)
            img = img.resize((128,96))
            self.outImg_allEig = ImageTk.PhotoImage(img)
            self.outputCanvas4.create_image(64,48,image=self.outImg_allEig)

            self.test_canvas.delete(self.bg)

        else:
            print("The number of handles is not 13.")
        for i in range(13):
            #createOvalEZ(self.test_canvas, self.handles[i][0][0], self.handles[i][0][1], 5, "red", "handleMark")
            createOvalEZ(self.test_canvas, newHandles_wv[i][0][0]*10-325+self.eyeCenterPos[0], newHandles_wv[i][0][1]*10-245+self.eyeCenterPos[1], 5, "green", "handleMark")
            if self.mode.get() == 1:
                createOvalEZ(self.test_canvas, testHandles[i][0][0]*10, testHandles[i][0][1]*10, 5, "blue", "handleMark")

        self.updatePreview()

    def genTest(self):
        errorArray = []
        for i in range(109,143):
            path = 'json_data/' + str(i+1).zfill(3) + '_v2.json'
            if not os.path.isfile(path):
                continue
            self.testInd.set(i+1)
            self.updateTestImg()
            self.save_canvas()
            errorArray.append(self.eyeErrors)
        errorArray = np.round(errorArray, decimals=4)
        #errorArray = sorted(errorArray, key=lambda x: x[1])
        errorArray = np.array(errorArray)
        avgErr = np.mean(errorArray,axis=0)
        avgErr = np.round(avgErr,decimals=4)
        stdErr = np.std(errorArray,axis=0)
        stdErr = np.round(stdErr,decimals=4)
        
        path_w = 'test_result/result.txt'


        with open(path_w, mode='w') as f:
            s = 'testNumber&RGB&psnrRGB&Lab&psnrL&psnrab&outl&anchorPix&anchorR\\\\ \\hline\n'
            s = 'testNumber&Lab&anchor&outl\\\\ \\hline\n'
            f.write(s)
            for err in errorArray:
                s1 = r"\begin{minipage}{0.15\linewidth}"
                s2 = r"\centering"
                s3 = r"\includegraphics[width=\linewidth]{./fig/test"+str(int(err[0]))+r".png}"
                s4 = r"\end{minipage} &"
                ss1 = r"\begin{minipage}{0.15\linewidth}"
                ss2 = r"\centering"
                ss3 = r"\includegraphics[width=\linewidth]{./fig/outline/test"+str(int(err[0]))+r".png}"
                ss4 = r"\end{minipage} &"
                #s5 = str(err[1]) + '&' + str(err[2]) + '&' + str(err[3]) + '&' + str(err[4]) + '&' + str(err[5]) + '&' + str(err[6]) + '&' + str(err[7]) + '&' + str(err[8]) + '\\\\\n'
                s5 = str(err[1]) + '&' + str(err[2]) + '&' + str(err[3]) + '\\\\\n'
                s = s1 + '\n' + s2 + '\n' + s3 + '\n' + s4 + '\n' + ss1 + '\n' + ss2 + '\n' + ss3 + '\n' + ss4 + '\n' + s5
                f.write(s)
            #s = 'avg&'+str(avgErr[1])+'&'+str(avgErr[2])+'&'+str(avgErr[3])+'&'+str(avgErr[4])+'&'+str(avgErr[5])+'&'+str(avgErr[6])+'&'+str(avgErr[7])+'&'+str(avgErr[8])+'\\\\\n'
            s = 'avg&'+str(avgErr[1])+'&'+str(avgErr[2])+'&'+str(avgErr[3])+'\\\\\n'
            f.write(s)
            #s = 'std&'+str(stdErr[1])+'&'+str(stdErr[2])+'&'+str(stdErr[3])+'&'+str(stdErr[4])+'&'+str(stdErr[5])+'&'+str(stdErr[6])+'&'+str(stdErr[7])+'&'+str(stdErr[8])+'\\\\\n'
            s = 'std&'+str(stdErr[1])+'&'+str(stdErr[2])+'&'+str(stdErr[3])+'\\\\\n'
            f.write(s)
    
    def paint(self, event): #色と太さを選んで絵をかく
        if self.mode.get() == 0:
            if self.eraser_on:
                paint_color = 'white'
            else:
                paint_color = colorCord(self.colR.get(), self.colG.get(), self.colB.get())
            if self.old_x and self.old_y:
                self.test_canvas.create_line(self.old_x, self.old_y, event.x, event.y, width=self.thickness.get(), fill=paint_color, capstyle=tkinter.ROUND, smooth=tkinter.TRUE, splinesteps=36)
                #self.draw.line((self.old_x, self.old_y, event.x, event.y), fill=paint_color, width=self.thickness.get())
            self.old_x = event.x
            self.old_y = event.y

    def register(self, event): #ハンドルを登録 指定順(白目上右下左, 黒目左, 黒目右, 白目右下, 白目左下, まつ毛左上（黒目左の真上くらい）, まつ毛右上（左上と同様）, 黒目右下, 黒目左下, 瞳孔)
        if self.mode.get() == 1:
            if self.handleNum < 13:

                self.handlePosCanvas.delete(tkinter.ALL)
                url = 'handlePos/handlePos'+str(self.handleNum+1 +1)+'.png'
                img = Image.open(url)
                img = img.resize((64,48))
                self.handlePosImg = ImageTk.PhotoImage(img)
                self.handlePosCanvas.create_image(32,24,image=self.handlePosImg)

                self.handles.append([[event.x-2, event.y-2]])
                createOvalEZ(self.test_canvas, event.x, event.y, 5, "red", "handleMark")
                print(event.x-2, event.y-2)
                self.handleNum += 1

    def register_RDL(self, event): #empcaで予測するときのハンドルを登録（right, down, left）
        if self.mode.get() == 1:
            if self.handleNum < 3:
                self.handles.append([[event.x-2, event.y-2]])
                createOvalEZ(self.test_canvas, event.x, event.y, 5, "red", "handleMark")
                print(event.x-2, event.y-2)
                self.handleNum += 1

    def setEyeCenterPos(self, event):
        if self.mode.get() != 1:
            self.test_canvas.delete("eyeCenter")
            self.eyeCenterPos = [event.x-2, event.y-2]
            createOvalEZ(self.test_canvas, event.x, event.y, 5, "green", "eyeCenter")
        else:
            center = pre.handlesArr[self.testInd.get()-1][12][0]
            self.eyeCenterPos = [center[0], center[1]]
            createOvalEZ(self.test_canvas, center[0], center[1], 5, "green", "eyeCenter")



    def reset(self, event):
        self.old_x, self.old_y = None, None

    def updateRefImg(self): # リファレンス画像を選んで更新
        self.refCanvas.delete(tkinter.ALL)
        ind = self.refInd.get()
        url = 'data_eyes_test/'+ str(ind).zfill(3)+ '.png'
        img = Image.open(url)
        self.refImg = ImageTk.PhotoImage(img)
        self.refCanvas.create_image(32,24,image=self.refImg)

    def updateRefImg2(self):
        self.refCanvas2.delete(tkinter.ALL)
        ind = self.refInd2.get()
        url = 'data_eyes_test/'+ str(ind).zfill(3)+ '.png'
        img = Image.open(url)
        self.refImg2 = ImageTk.PhotoImage(img)
        self.refCanvas2.create_image(0, 0, image=self.refImg2, anchor='nw')

    def updateTestImg(self): # テスト画像を選んで更新
        self.test_canvas.delete(tkinter.ALL)
        ind = self.testInd.get()
        url = 'data_eyes_test/'+ str(ind).zfill(3)+ '.png'
        #url = 'data_eyes_test/roughSketchA.png'
        img = Image.open(url)
        img = img.resize((640,480))
        self.loadImg = ImageTk.PhotoImage(img)
        self.test_canvas.create_image(0, 0, image=self.loadImg, anchor=tkinter.NW)

    def updatePreview(self, event=None):
        eyeL = cv2.imread('output/outputEyeImg.png')
        eyeR = cv2.flip(eyeL, 1)
        if(self.previewEyeR.get()):
            back = cv2.imread('preview/'+str(self.previewInd.get())+'_noLEye.png')
        else:
            back = cv2.imread('preview/'+str(self.previewInd.get())+'_noEye.png')
        maskL = cv2.imread('temp_img/eyeMask.png')
        maskR = cv2.flip(maskL, 1)

        previewImg = blendPreview(eyeL, back, maskL, self.previewEyeLCenter, self.previewEyeScale.get()/100)
        previewImg = blendPreview(eyeR, previewImg, maskR, self.previewEyeRCenter, self.previewEyeScale.get()/100)
        cv2.imwrite('temp_img/preview.png',previewImg)

        image_rgb = cv2.cvtColor(previewImg, cv2.COLOR_BGR2RGB) # imreadはBGRなのでRGBに変換
        image_pil = Image.fromarray(image_rgb) # RGBからPILフォーマットへ変換
        self.previewImgTk  = ImageTk.PhotoImage(image_pil) # ImageTkフォーマットへ変換
        self.previewCanvas.create_image(0, 0, image=self.previewImgTk, anchor='nw')

    def previewB1Motion(self, event):
        self.previewEyeLCenter = (event.x, event.y)
        self.previewEyeRCenter = (300-event.x, event.y)
        self.updatePreview()
        #print("previewB1Motion")

    def addDetails(self):
        tmpDiff = 1000000000
        handleIndex = -1
        for i in range(109):#,len(pre.handlesArr)):
            pixelDiff, ratioDiff, boundingBoxLen = calcHandleDiff(self.outputHandles, pre.handlesArr[i])
            if pixelDiff < tmpDiff:
                tmpDiff = pixelDiff
                handleIndex = i
        #self.refInd.set(handleIndex+1) # 詳細の参照画像を自由に選びたいときは消す
        self.updateRefImg()
        #newEye_TC = transferColor('output/'+self.textName.get()+'_wv_output.png', self.outputHandles, 'data_eyes/'+str(self.refInd.get()).zfill(3)+'.png', pre.handlesArr[self.refInd.get()-1])
        newEye_TC = transferColor('data_eyes_test/'+str(self.refInd.get()).zfill(3)+'.png', pre.handlesArr[self.refInd.get()-1], 'output/'+self.textName.get()+'_wv_output.png', self.outputHandles)
        cv2.imwrite('output/outputEyeImg.png', newEye_TC)
        cv2.imwrite('output/detailedEyeImg.png', newEye_TC)

        self.outputCanvas2.delete(tkinter.ALL)
        url = 'output/outputEyeImg.png'
        img = Image.open(url)
        img = img.resize((128, 96))
        self.outImg2 = ImageTk.PhotoImage(img)
        self.outputCanvas2.create_image(64,48,image=self.outImg2)

        newEye_wv, newHandles_wv, newEyeMask_wv , _err= project_withVector(newEye_TC, self.outputHandles,contRate=self.contRate.get(),refs=self.refs)
        #newEye_wv = cv2.bilateralFilter(newEye_wv, 3, 200, 100)
        cv2.imwrite('output/outputEyeImg.png', newEye_wv)
        cv2.imwrite('output/reprojectedEyeImg.png', newEye_wv)
        cv2.imwrite('temp_img/reprojectedEyeImg.png', newEye_wv)
        cv2.imwrite('temp_img/eyeMask.png', newEyeMask_wv)

        resultImg = np.append(self.normalizedSketch,newEye_wv,axis=1)
        cv2.imwrite('temp_img/sketchAndResult.png', resultImg)


        self.outputCanvas3.delete(tkinter.ALL)
        url = 'output/outputEyeImg.png'
        img = Image.open(url)
        img = img.resize((128, 96))
        self.outImg3 = ImageTk.PhotoImage(img)
        self.outputCanvas3.create_image(64,48,image=self.outImg3)

        self.updatePreview()

    """
    def generateHandle(self):
        self.test_canvas.delete("handleMark")

        self.test_canvas.postscript(file='temp_img/out.ps', colormode='color')
        psimage=Image.open('temp_img/out.ps')

        psimage.save('temp_img/out.png')

        pngImg = cv2.imread('temp_img/out.png')
        print(pngImg.shape)
        pngImg = pngImg[2:362, 2:482]
        pngImg = cv2.resize(pngImg, (640,480))

        dx = 325-self.eyeCenterPos[0]
        dy = 245-self.eyeCenterPos[1]
        afin_matrix = np.float32([[1,0,dx],[0,1,dy]])
        pngImg = cv2.warpAffine(pngImg, afin_matrix, (640,480), borderValue = (255,255,255))

        sketch = cv2.resize(pngImg, (64,48))
        sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)

        edge = findEdge(sketch)
        cv2.drawMarker(sketch, (int(edge["R"][0]), int(edge["R"][1])), (0,0,0))
        cv2.drawMarker(sketch, (int(edge["L"][0]), int(edge["L"][1])), (0,0,0))
        cv2.drawMarker(sketch, (int(edge["D"][0]), int(edge["D"][1])), (0,0,0))
        cv2.imwrite('temp_img/autoHandleGen.png', sketch)
        
        edgeR = np.array(edge["R"])
        edgeD = np.array(edge["D"])
        edgeL = np.array(edge["L"])
        
        edgeR = np.array([edgeR])
        edgeD = np.array([edgeD])
        edgeL = np.array([edgeL])

        handles = np.array([edgeR,edgeD,edgeL,np.array([[32.5,24.5]])])

        newImgVec, newHandleVec = project_autoHandleGen(sketch,handles)
        newHandleVec = newHandleVec.reshape(13,1,2)
        newHandleVec = newHandleVec*10 - np.array([[dx, dy]])
        self.handles = newHandleVec
        self.handleNum = 13

        for i in range(13):
            createOvalEZ(self.test_canvas, newHandleVec[i][0][0]-dx, newHandleVec[i][0][1]-dy, 5, "red", "handleMark")
        
        newImg = newImgVec.reshape(48,64)
        newImg = np.clip(newImg, 0, 255)
        newImg = newImg.astype(np.uint8)
        #cv2.imshow("image", newImg)
        #cv2.waitKey()
    """

    def generateHandle_EM(self):
        
        self.test_canvas.delete("handleMark")

        self.test_canvas.postscript(file='temp_img/out.ps', colormode='color')
        psimage=Image.open('temp_img/out.ps')

        psimage.save('temp_img/out.png')

        pngImg = cv2.imread('temp_img/out.png')
        print(pngImg.shape)
        pngImg = pngImg[2:362, 2:482]
        pngImg = cv2.resize(pngImg, (640,480))

        dx = 325-self.eyeCenterPos[0]
        dy = 245-self.eyeCenterPos[1]
        afin_matrix = np.float32([[1,0,dx],[0,1,dy]])
        pngImg = cv2.warpAffine(pngImg, afin_matrix, (640,480), borderValue = (255,255,255))

        sketch = cv2.resize(pngImg, (64,48))
        self.normalizedSketch = sketch
        cv2.imwrite('temp_img/normalizedSketch.png', sketch)
        sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)
        
        if self.mode.get() == 0:
            edge = findEdge(sketch)

            sketchMark = np.copy(sketch)
            cv2.drawMarker(sketchMark, (int(edge["R"][0]), int(edge["R"][1])), (0,0,0))
            cv2.drawMarker(sketchMark, (int(edge["L"][0]), int(edge["L"][1])), (0,0,0))
            cv2.drawMarker(sketchMark, (int(edge["D"][0]), int(edge["D"][1])), (0,0,0))
            cv2.imwrite('temp_img/autoHandleGen.png', sketchMark)
            
            edgeR = np.array(edge["R"])
            edgeD = np.array(edge["D"])
            edgeL = np.array(edge["L"])
            
            edgeR = np.array([edgeR])
            edgeD = np.array([edgeD])
            edgeL = np.array([edgeL])

            if self.generateNum == 0:
                self.sketcHhandles = np.array([edgeR,edgeD,edgeL,np.array([[32.5,24.5]])])
            handles = self.sketcHhandles
        elif self.mode.get() == 1:
            truehand = pre.handlesArr[self.testInd.get()-1]
            self.handles = []
            self.handles.append(truehand[1])
            self.handles.append(truehand[2])
            self.handles.append(truehand[3])
            self.handles.append(truehand[12])
            self.handles = np.array(self.handles)
            handles = self.handles

        self.bg = self.test_canvas.create_rectangle(-1000, -1000, 1000, 1000, fill = 'linen')
        self.test_canvas.tag_lower(self.bg)
        self.test_canvas.postscript(file='temp_img/out.ps', colormode='color')
        psimage=Image.open('temp_img/out.ps')

        psimage.save('temp_img/out.png')

        pngImg = cv2.imread('temp_img/out.png')
        print(pngImg.shape)
        pngImg = pngImg[2:362, 2:482]
        pngImg = cv2.resize(pngImg, (640,480))

        dx = 325-self.eyeCenterPos[0]
        dy = 245-self.eyeCenterPos[1]
        afin_matrix = np.float32([[1,0,dx],[0,1,dy]])
        pngImg = cv2.warpAffine(pngImg, afin_matrix, (640,480), borderValue = (230,240,250))

        sketch_bg = cv2.resize(pngImg, (64,48))
        self.normalizedSketch = sketch_bg
        cv2.imwrite('temp_img/normalizedSketch.png', sketch_bg)
        sketch_bg = cv2.cvtColor(sketch_bg, cv2.COLOR_BGR2GRAY)

        newImgVec, newHandleVec = project_autoHandleGen_EM(sketch_bg,handles)
        #print("handleTest",newHandleVec, handles)
        newHandleVec = newHandleVec.reshape(13,1,2)
        newHandleVec = newHandleVec*10 - np.array([[dx, dy]])
        self.handles = newHandleVec
        self.handleNum = 13

        for i in range(13):
            createOvalEZ(self.test_canvas, newHandleVec[i][0][0], newHandleVec[i][0][1], 5, "red", "handleMark")
        
        newImg = newImgVec.reshape(48,64)
        newImg = np.clip(newImg, 0, 255)
        newImg = newImg.astype(np.uint8)
        #cv2.imshow("image", newImg)
        #cv2.waitKey()

        self.generateNum += 1



root = tkinter.Tk()
app = Application(master=root)
app.mainloop()