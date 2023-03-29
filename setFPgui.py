from distutils.cmd import Command
from statistics import variance
import tkinter
from tkinter import Variable, ttk
from turtle import color
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

def colorCord(R,G,B):
    return('#'+ format(R, '02x')+ format(G, '02x')+ format(B, '02x'))


class Application(tkinter.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title('eye_projecter')
        self.pack()
        self.create_widgets()
        self.setup()
        self.handles = np.load('saves/HD_handlesList_1.npy')
        self.existHandles = np.load('saves/HD_existHandles_1.npy')
        self.exceptList = np.load('saves/HD_exceptionList_1.npy')
        self.pointRad = 3

        print(self.handles[0][0][0], self.existHandles[0][0], self.exceptList.shape)

    def create_widgets(self):
        self.vr = tkinter.IntVar()
        self.vr.set(1)
        self.write_radio = tkinter.Radiobutton(self, text='write', variable=self.vr, value=1, command=self.change_radio)
        self.write_radio.grid(row=0, column=0)
        self.erase_radio = tkinter.Radiobutton(self, text='erase', variable=self.vr, value=2, command=self.change_radio)
        self.erase_radio.grid(row=0, column=1)

        self.clear_button = tkinter.Button(self, text='clear all', command=self.clear_canvas)
        self.clear_button.grid(row=0, column=2)

        self.clear_button = tkinter.Button(self, text='save', command=self.save)
        self.clear_button.grid(row=0, column=3)

        self.exceptionButton = tkinter.Button(self, text='except', command=self.exception)
        self.exceptionButton.grid(row=1, column=0)

        self.exceptText = tkinter.Label(self, text='', fg="black")
        self.exceptText.grid(row=1, column=1)

        self.imgInd = tkinter.IntVar()
        self.imgInd.set(1)
        self.selImgInd = tkinter.Spinbox(self, from_=1, to=143, increment=1, textvariable=self.imgInd, command=self.dispImg)
        self.selImgInd.grid(row=2, column=3)

        self.coordinateText = tkinter.Label(self, text='0,0')
        self.coordinateText.grid(row=2, column=2)

        self.isShowAll = tkinter.BooleanVar()
        self.checkShowAll = tkinter.Checkbutton(self, text = "show all", variable=self.isShowAll)
        self.checkShowAll.grid(row=2, column=0)

        self.handleInd = tkinter.IntVar()
        self.handleInd.set(0)
        self.selhandleInd = tkinter.Spinbox(self, from_=0, to=99, increment=1, textvariable=self.handleInd, command=self.dispImg)
        self.selhandleInd.grid(row=2, column=1)

        self.test_canvas = tkinter.Canvas(self, bg='white', width=640, height=480)
        self.test_canvas.grid(row=3, column=0, columnspan=4)
        self.test_canvas.bind('<Button-1>', self.getCoordinate)
        self.test_canvas.bind('<B1-Motion>', self.getCoordinate)
        self.test_canvas.bind('<ButtonRelease-1>', self.reset)

        self.master.bind('<KeyPress>', self.keyHandler)

        url = 'data_eyes/'+ str(1).zfill(3)+ '.png'
        img = Image.open(url)
        img = img.resize((640, 480))
        self.photo_image = ImageTk.PhotoImage(img)

        # キャンバスのサイズを取得
        self.update() # Canvasのサイズを取得するため更新しておく
        canvas_width = self.test_canvas.winfo_width()
        canvas_height = self.test_canvas.winfo_height()

        # 画像の描画
        self.test_canvas.create_image(
                canvas_width / 2,       # 画像表示位置(Canvasの中心)
                canvas_height / 2,                   
                image=self.photo_image  # 表示画像データ
                )

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

    def clear_canvas(self):
        self.test_canvas.delete("all")
        self.handles = []
        self.handleNum = 0

    def getCoordinate(self, event):
        self.coordinateText["text"] = str(event.x-2) + ',' + str(event.y-2)
        #print("click")

        self.test_canvas.delete("point")
        self.test_canvas.create_oval(event.x-self.pointRad, event.y-self.pointRad, event.x+self.pointRad, event.y+self.pointRad, fill="red", tag = "point")

        self.existHandles[self.imgInd.get()-1][self.handleInd.get()] = 1
        self.handles[self.imgInd.get()-1][self.handleInd.get()][0] = np.array([event.x-2, event.y-2])

    def dispImg(self):
        if self.exceptList[self.imgInd.get()-1] == 1:
            self.exceptText["text"] = "exception"
            self.exceptText["fg"] = "red"
        else:
            self.exceptText["text"] = "not exception"
            self.exceptText["fg"] = "black"
        
        self.test_canvas.delete(tkinter.ALL)

        ind = self.imgInd.get()
        url = 'data_eyes/'+ str(ind).zfill(3)+ '.png'
        img = Image.open(url)
        img = img.resize((640, 480))
        self.photo_image = ImageTk.PhotoImage(img)


        self.update() # Canvasのサイズを取得するため更新しておく
        canvas_width = self.test_canvas.winfo_width()
        canvas_height = self.test_canvas.winfo_height()

        # 画像の描画
        self.test_canvas.create_image(canvas_width / 2, canvas_height / 2, image=self.photo_image)

        if self.isShowAll.get():
            for i in range(100):
                if self.existHandles[self.imgInd.get()-1][i] == 0:
                    break
                elif i == self.handleInd.get():
                    pos = self.handles[self.imgInd.get()-1][i][0]
                    self.test_canvas.create_oval(pos[0]+2-self.pointRad, pos[1]+2-self.pointRad, pos[0]+2+self.pointRad, pos[1]+2+self.pointRad, fill="red", tag = "point")
                elif i == 0 or i == 4 or i == 7:
                    pos = self.handles[self.imgInd.get()-1][i][0]
                    self.test_canvas.create_oval(pos[0]+2-self.pointRad, pos[1]+2-self.pointRad, pos[0]+2+self.pointRad, pos[1]+2+self.pointRad, fill="blue", tag = "pointG")
                else:
                    pos = self.handles[self.imgInd.get()-1][i][0]
                    self.test_canvas.create_oval(pos[0]+2-self.pointRad, pos[1]+2-self.pointRad, pos[0]+2+self.pointRad, pos[1]+2+self.pointRad, fill="gray", tag = "pointG")
        else:
            if self.existHandles[self.imgInd.get()-1][self.handleInd.get()] == 1:
                pos = self.handles[self.imgInd.get()-1][self.handleInd.get()][0]
                self.test_canvas.create_oval(pos[0]+2-self.pointRad, pos[1]+2-self.pointRad, pos[0]+2+self.pointRad, pos[1]+2+self.pointRad, fill="red", tag = "point")

    def dispHandle(self):
        if self.existHandles[self.imgInd.get()-1][self.handleInd.get()] == 1:
            pos = self.handles[self.imgInd.get()-1][self.handleInd.get()][0]
            self.test_canvas.create_oval(pos[0]+2-self.pointRad, pos[1]+2-self.pointRad, pos[0]+2+self.pointRad, pos[1]+2+self.pointRad, fill="red", tag = "point")

    def keyHandler(self, event):
        print(event.keycode)
        key = event.keycode
        if key == 68 and self.imgInd.get()<143:
            self.imgInd.set(self.imgInd.get()+1)
        if key == 65 and self.imgInd.get()>1:
            self.imgInd.set(self.imgInd.get()-1)

        if key == 87 and self.handleInd.get()<99:
            self.handleInd.set(self.handleInd.get()+1)
        if key == 83 and self.handleInd.get()>0:
            self.handleInd.set(self.handleInd.get()-1)
        
        self.dispImg()

    def exception(self):
        self.exceptList[self.imgInd.get()-1] = 1 - self.exceptList[self.imgInd.get()-1]

        if self.exceptList[self.imgInd.get()-1] == 1:
            self.exceptText["text"] = "exception"
            self.exceptText["fg"] = "red"
        else:
            self.exceptText["text"] = "not exception"
            self.exceptText["fg"] = "black"

    def save(self):
        print(self.handles[0][0][0], self.existHandles[0][0], self.exceptList.shape)
        np.save('saves/HD_handlesList_1', self.handles)
        np.save('saves/HD_existHandles_1', self.existHandles)
        np.save('saves/HD_exceptionList_1', self.exceptList)
        print("saved.")


    def reset(self, event):
        self.old_x, self.old_y = None, None

root = tkinter.Tk()
app = Application(master=root)
app.mainloop()