import cv2
import numpy as np

from mesh2Lib import mesh2
import preData as pre

def transferColor(pngURL_org, handles_org, pngURL_ref, handles_ref):
    orgImg = cv2.imread(pngURL_org)
    refImg = cv2.imread(pngURL_ref)

    orgEyeMesh = mesh2(64,48, orgImg)
    orgEyeMesh.setHandlesOrg(handles_org)
    orgEyeMesh.setHandlesDfm(pre.handlesAvg)
    orgEyeMesh.applyHandles()
    orgImgNormalized = orgEyeMesh.deform()

    refEyeMesh = mesh2(64,48, refImg)
    refEyeMesh.setHandlesOrg(handles_ref)
    refEyeMesh.setHandlesDfm(pre.handlesAvg)
    refEyeMesh.applyHandles()
    refImgNormalized = refEyeMesh.deform()

    orgHSV = cv2.cvtColor(orgImgNormalized,cv2.COLOR_BGR2HSV)
    refHSV = cv2.cvtColor(refImgNormalized,cv2.COLOR_BGR2HSV)

    print(orgHSV[47][63])

    for i in range(48):
        for j in range(64):
            orgHSV[i][j][0] = refHSV[i][j][0]
            #orgHSV[i][j][1] = refHSV[i][j][1]

    rtnImgNormalized = cv2.cvtColor(orgHSV,cv2.COLOR_HSV2BGR)

    rtnEyeMesh = mesh2(64,48, rtnImgNormalized)
    rtnEyeMesh.setHandlesOrg(pre.handlesAvg)
    rtnEyeMesh.setHandlesDfm(handles_ref)
    rtnEyeMesh.applyHandles()
    rtnImg = rtnEyeMesh.deform()

    return rtnImg

def createOvalEZ(canvas, X, Y, rad, color, tag):
    canvas.create_oval(X-rad, Y-rad, X+rad, Y+rad, fill=color, tag = tag, width=0)

def isExeption(i): 
    return (i == 84 or i == 122 or i == 123)

orgInd = 17
refInd = 21
img = transferColor('data_eyes/'+str(orgInd).zfill(3)+'.png', pre.handlesArr[orgInd-1], 'data_eyes/'+str(refInd).zfill(3)+'.png', pre.handlesArr[refInd-1])

#cv2.imshow("Image", img)
#cv2.waitKey()



