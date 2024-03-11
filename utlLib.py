import cv2
import numpy as np
import matplotlib.pyplot as plt

from mesh2Lib import mesh2
import preData as pre

def gradientDescent(func, init, alpha, maxitr=1000, tol=1.0e-4):
    x = init
    for itr in range(maxitr):
        if (np.abs(alpha*func(x)) < tol).all():
            return x
        x -= alpha*func(x)
    print("Not converged.")
    return x
    #raise ValueError('Not converged.')


def blendPreview(eye, back, mask, center=(185, 144), k=1): #eye,maskは(48, 64, 3) backは(300,300,3) cv2.imreadしてから引数にしてください
    mask = mask * 255

    resizeMask = cv2.resize(mask, None, fx = k, fy = k)
    resizeEye = cv2.resize(eye, None, fx = k, fy = k)

    size = resizeMask.shape

    maskBig = np.zeros((300,300,3), np.uint8)
    eyeBig = np.zeros((300,300,3), np.uint8)

    maskBig[0:size[0], 0:size[1]] = resizeMask
    eyeBig[0:size[0], 0:size[1]] = resizeEye

    result = cv2.seamlessClone(eyeBig, back, maskBig, center, cv2.NORMAL_CLONE)

    return result

def transferColor(pngURL_ref, handles_ref, pngURL_org, handles_org, isOutNormal = False):
    orgImg = cv2.imread(pngURL_org)
    refImg = cv2.imread(pngURL_ref)

    orgEyeMesh = mesh2(64,48, orgImg)
    orgEyeMesh.setHandlesOrg(handles_org)
    orgEyeMesh.setHandlesDfm(pre.handlesAvg)
    orgEyeMesh.applyHandles()
    orgImgNormalized = orgEyeMesh.deform(whiteback=True)
    cv2.imwrite('temp_img/detailingOrgNormalized.png', orgImgNormalized)

    refEyeMesh = mesh2(64,48, refImg)
    refEyeMesh.setHandlesOrg(handles_ref)
    refEyeMesh.setHandlesDfm(pre.handlesAvg)
    refEyeMesh.applyHandles()
    refImgNormalized = refEyeMesh.deform(whiteback=True)
    cv2.imwrite('temp_img/detailingRefNormalized.png', refImgNormalized)

    orgHSV = cv2.cvtColor(orgImgNormalized,cv2.COLOR_BGR2HSV)
    refHSV = cv2.cvtColor(refImgNormalized,cv2.COLOR_BGR2HSV)

    for i in range(48):
        for j in range(64):
            orgHSV[i][j][2] = refHSV[i][j][2]
            #orgHSV[i][j][1] = refHSV[i][j][1]

    rtnImgNormalized = cv2.cvtColor(orgHSV,cv2.COLOR_HSV2BGR)

    rtnEyeMesh = mesh2(64,48, rtnImgNormalized)
    rtnEyeMesh.setHandlesOrg(pre.handlesAvg)
    rtnEyeMesh.setHandlesDfm(handles_org)
    rtnEyeMesh.applyHandles()
    rtnImg = rtnEyeMesh.deform(whiteback=True)

    if isOutNormal:
        return rtnImgNormalized
    else:
        return rtnImg
    
def transferColor_normal(refImgNormalized, orgImgNormalized):

    orgHSV = cv2.cvtColor(orgImgNormalized,cv2.COLOR_BGR2HSV)
    refHSV = cv2.cvtColor(refImgNormalized,cv2.COLOR_BGR2HSV)

    for i in range(48):
        for j in range(64):
            orgHSV[i][j][2] = refHSV[i][j][2]
            #orgHSV[i][j][1] = refHSV[i][j][1]

    rtnImgNormalized = cv2.cvtColor(orgHSV,cv2.COLOR_HSV2BGR)

    return rtnImgNormalized

def createOvalEZ(canvas, X, Y, rad, color, tag):
    canvas.create_oval(X-rad, Y-rad, X+rad, Y+rad, fill=color, tag = tag, width=0)

def isExeption(i): 
    return (i == 84 or i == 122 or i == 123)

def calcHandleDiff(handleTrue, handle): #handleはnp.array (13,1,2)の形
    diff = handleTrue - handle
    pixelDiff = 0
    handleNum = 0
    for d in diff:
        handleNum+= 1
        pixelDiff += np.linalg.norm(d[0], ord=2)
    pixelDiff = pixelDiff/handleNum

    boundingBoxLen = handleBoundingBox(handleTrue)
    ratioDiff = pixelDiff/boundingBoxLen

    return pixelDiff, ratioDiff, boundingBoxLen

def handleBoundingBox(handle):
    minX = 100000
    maxX = 0
    minY = 100000
    maxY = 0
    for h in handle:
        if h[0][0] < minX:
            minX = h[0][0]
        if h[0][1] < minY:
            minY = h[0][1]
        if h[0][0] > maxX:
            maxX = h[0][0]
        if h[0][1] > maxY:
            maxY = h[0][1]
    min = np.array([minX, minY])
    max = np.array([maxX, maxY])

    return np.linalg.norm(max-min, ord=2)

orgInd = 17
refInd = 21
img = transferColor('data_eyes/'+str(orgInd).zfill(3)+'.png', pre.handlesArr[orgInd-1], 'data_eyes/'+str(refInd).zfill(3)+'.png', pre.handlesArr[refInd-1])

#cv2.imshow("Image", img)
#cv2.waitKey()



