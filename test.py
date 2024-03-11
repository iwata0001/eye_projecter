import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import preData2
from mesh2Lib import mesh2
# test2

def main():
    eigVal = np.load('saves/10000data_eigval.npy')
    eigVec = np.load('saves/10000data_eigvec.npy')
    print("eigvalshape",eigVal.shape)

    avgImgV, avgHandlesV, avgVectorV = np.split(preData2.avgdata_v1, [48*64*3, 48*64*3+13*1*2])

    avgImg = avgImgV / preData2.eyeCoeff
    avgImg = avgImg.reshape(48,64,3)
    avgHandles = avgHandlesV / preData2.handCoeff_v1
    avgHandles = avgHandles.reshape(13,1,2)

    cv2.imwrite('C:/pics/add/x_avgEye.png', avgImg)

    eigNums = [0, 1, 5, 29, 69]
    eigNums = range(100)
    c1 = 2
    MPdiffs = []

    path_w = 'stds.txt'
    s = ''

    for eigNum in eigNums:
        c2 = np.sqrt(eigVal[eigNum])
        s = s+str(c2)+'\t'
        # eigNum番目の固有値を足す
        dfmDataP = preData2.avgdata_v1 + c1*c2*eigVec[eigNum]
        dfmImgV, dfmHandlesV, dfmVectorV = np.split(dfmDataP, [48*64*3, 48*64*3+13*1*2])

        dfmImg = dfmImgV / preData2.eyeCoeff
        dfmImg = dfmImg.reshape(48,64,3)
        dfmImg = np.clip(dfmImg, 0, 255)
        dfmHandles = dfmHandlesV / preData2.handCoeff_v1
        dfmHandles = dfmHandles.reshape(13,1,2)

        eyeMesh = mesh2(64,48, dfmImg)
        eyeMesh.setHandlesOrg(avgHandles)
        eyeMesh.setHandlesDfm(dfmHandles)
        eyeMesh.applyHandles()

        dfmImgP = eyeMesh.deform()

        cv2.imwrite('C:/pics/add/dfmEye'+str(eigNum)+'P.png', dfmImgP)

        # eigNum番目の固有値を引く
        dfmDataP = preData2.avgdata_v1 - c1*c2*eigVec[eigNum]
        dfmImgV, dfmHandlesV, dfmVectorV = np.split(dfmDataP, [48*64*3, 48*64*3+13*1*2])

        dfmImg = dfmImgV / preData2.eyeCoeff
        dfmImg = dfmImg.reshape(48,64,3)
        dfmImg = np.clip(dfmImg, 0, 255)
        dfmHandles = dfmHandlesV / preData2.handCoeff_v1
        dfmHandles = dfmHandles.reshape(13,1,2)

        eyeMesh = mesh2(64,48, dfmImg)
        eyeMesh.setHandlesOrg(avgHandles)
        eyeMesh.setHandlesDfm(dfmHandles)
        eyeMesh.applyHandles()

        dfmImgM = eyeMesh.deform()

        cv2.imwrite('C:/pics/add/dfmEye'+str(eigNum)+'M.png', dfmImgM)

        w = 64
        h = 48
        dfmImgMfloat = dfmImgM.astype(np.float32)
        dfmImgPfloat = dfmImgP.astype(np.float32)
        MPdiff = dfmImgMfloat - dfmImgPfloat
        #MPdiff = cv2.resize(MPdiff,(w,h))
        #MPdiff = cv2.blur(MPdiff,(32,32))

        grayImg = np.zeros_like(MPdiff) + 128
        MPdiffImg = grayImg + MPdiff/2
        MPdiffImg = MPdiffImg.astype(np.uint8)
        MPdiffImg = np.clip(MPdiffImg,0,255)
        cv2.imwrite('C:/pics/MPdiffs/'+str(eigNum)+'.png', MPdiffImg)

        MPdiffV = MPdiff.reshape(h*w*3)
        MPdiffNorm = np.linalg.norm(MPdiffV,ord=2)
        MPdiffs.append(MPdiffNorm)

        resultImg = np.append(dfmImgM,avgImg,axis=1)
        resultImg = np.append(resultImg,dfmImgP,axis=1)
        cv2.imwrite('C:/pics/eigVariation/variation'+str(eigNum)+'.png', resultImg)
    print(MPdiffs)

    with open(path_w, mode='w') as f:
        f.write(s)

if __name__ == '__main__':
    main()