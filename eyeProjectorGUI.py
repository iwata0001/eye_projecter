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
from PIL import ImageTk

from projectInput import project, project_addDetail
from projectInput_vector import project_withVector
#from eyeVectorizerGUI import Application_EVGUI
import DmeshLib as DMesh

def colorCord(R,G,B): #整数値RGBをカラーコードに
    return('#'+ format(R, '02x')+ format(G, '02x')+ format(B, '02x'))


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

        self.thickness = tkinter.IntVar()
        self.thickness.set(1)
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
        self.mode.set(0)
        self.modeSelect = tkinter.Scale(self, label='mode', from_=0, to=1, orient=tkinter.HORIZONTAL, variable=self.mode)
        self.modeSelect.grid(row=2, column=0)

        self.textName = tkinter.Entry(self, width=20)
        self.textName.grid(row=2, column=1)

        self.refCanvas = tkinter.Canvas(self, bg='white', width = 64, height=48)
        self.refCanvas.grid(row=3, column=0)

        self.refInd = tkinter.IntVar()
        self.refInd.set(1)
        self.selRefInd = tkinter.Spinbox(self, from_=1, to=143, increment=1, textvariable=self.refInd, command=self.updateRefImg)
        self.selRefInd.grid(row=3, column=1)

        self.detailButton = tkinter.Button(self, text='add details', command=self.addDetails)
        self.detailButton.grid(row=3, column=2)

        ind = self.refInd.get()
        url = 'data_eyes/'+ str(ind).zfill(3)+ '.png'
        img = Image.open(url)
        self.refImg = ImageTk.PhotoImage(img)
        self.refCanvas.create_image(32,24,image=self.refImg)

        self.test_canvas = tkinter.Canvas(self, bg='white', width=640, height=480)
        self.test_canvas.grid(row=4, column=0, columnspan=4)
        self.test_canvas.bind('<B1-Motion>', self.paint)
        self.test_canvas.bind('<Button-1>', self.register)
        self.test_canvas.bind('<ButtonRelease-1>', self.reset)

        self.outputCanvas = tkinter.Canvas(self, bg='white', width=64, height=48)
        self.outputCanvas.grid(row=5, column=0)

        self.EVGUI = None

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

    def save_canvas(self): #入力を投影して出力を画像で保存
        if self.handleNum == 13:
            self.test_canvas.postscript(file='out.ps', colormode='color')
            psimage=Image.open('out.ps')

            print(psimage.size)
            psimage.save('out.png')

            pngImg = cv2.imread('out.png')
            pngImg = pngImg[2:362, 2:482]
            pngImg = cv2.resize(pngImg, (640,480))
            print(pngImg.shape)

            handles = np.array(self.handles)

            dx = 325 - handles[12][0][0]
            dy = 245 - handles[12][0][1]

            handles = handles + [[dx, dy]]
            handles = handles / 10

            afin_matrix = np.float32([[1,0,dx],[0,1,dy]])
            pngImg = cv2.warpAffine(pngImg, afin_matrix, (640,480))
            print(dx, dy)
            pngImg = cv2.resize(pngImg, (64,48))
            
            newEye = project(pngImg, handles)
            newEye_wv = project_withVector(pngImg, handles)
            cv2.imwrite('output/'+self.textName.get()+'_output.png', newEye)
            cv2.imwrite('output/'+self.textName.get()+'_input.png', pngImg)
            cv2.imwrite('output/'+self.textName.get()+'_wv_output.png', newEye_wv)

            self.outputCanvas.delete(tkinter.ALL)
            url = 'output/'+self.textName.get()+'_wv_output.png'
            img = Image.open(url)
            self.outImg = ImageTk.PhotoImage(img)
            self.outputCanvas.create_image(32,24,image=self.outImg)

        else:
            print("The number of handles is not 13.")
        


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
                self.handles.append([[event.x-2, event.y-2]])
                print(event.x-2, event.y-2)
                self.handleNum += 1


    def reset(self, event):
        self.old_x, self.old_y = None, None

    def updateRefImg(self): # リファレンス画像を選んで更新
        self.refCanvas.delete(tkinter.ALL)
        ind = self.refInd.get()
        url = 'data_eyes/'+ str(ind).zfill(3)+ '.png'
        img = Image.open(url)
        self.refImg = ImageTk.PhotoImage(img)
        self.refCanvas.create_image(32,24,image=self.refImg)

    def addDetails(self): # 描いた絵を入力にして, 投影したあとリファレンスの詳細を追加したものを表示 画像で保存
        self.test_canvas.postscript(file='out.ps', colormode='color')
        psimage=Image.open('out.ps')

        print(psimage.size)
        psimage.save('out.png')

        inputImg = cv2.imread('out.png')
        inputImg = inputImg[2:362, 2:482]
        inputImg = cv2.resize(inputImg, (64,48))

        plt.imshow(cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
        plt.show()

        inputHandles = np.array(self.handles)
        dx = 325 - inputHandles[12][0][0]
        dy = 245 - inputHandles[12][0][1]

        inputHandles = inputHandles + [[dx, dy]]
        inputHandles = inputHandles / 10

        ind = self.refInd.get()
        refImg = cv2.imread('data_eyes/'+ str(ind).zfill(3)+ '.png')

        plt.imshow(cv2.cvtColor(refImg, cv2.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
        plt.show()

        handles = DMesh.detectP('data_eyes_p/'+ str(ind).zfill(3) +'_p.png')
        handles2 = DMesh.detectP('data_eyes_p2/'+ str(ind).zfill(3) +'_p2.png')
        handles3 = DMesh.detectP('data_eyes_p3/'+ str(ind).zfill(3) +'_p3.png')

        refHandles = np.append(handles, handles2, axis=0)
        refHandles = np.append(refHandles, handles3, axis=0)
        refHandles = np.append(refHandles, np.array([[[32.5, 24.5]]]), axis=0)

        detailedImg = project_addDetail(inputImg, refImg, inputHandles, refHandles)

        plt.imshow(cv2.cvtColor(detailedImg, cv2.COLOR_BGR2RGB)) # OpenCV は色がGBR順なのでRGB順に並べ替える
        plt.show()

        cv2.imwrite('output/'+self.textName.get() + '+' + str(ind).zfill(3) + '_output.png', detailedImg)

        self.outputCanvas.delete(tkinter.ALL)
        url = 'output/'+self.textName.get() + '+' + str(ind).zfill(3) + '_output.png'
        img = Image.open(url)
        self.outImg = ImageTk.PhotoImage(img)
        self.outputCanvas.create_image(32,24,image=self.outImg)



root = tkinter.Tk()
app = Application(master=root)
app.mainloop()