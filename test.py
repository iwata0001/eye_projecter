import cv2
import numpy as np

def main():
    image = cv2.imread('pupil_tex/003.png')
    image = cv2.resize(image, (640, 480))

    # BGR-HSV変換
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

    # パラメータ
    region_size = 20
    ruler = 0.1
    min_element_size = 10
    num_iterations = 4

    # LSCインスタンス生成
    slc = cv2.ximgproc.createSuperpixelLSC(converted, region_size,float(ruler))
    slc.iterate(num_iterations)
    slc.enforceLabelConnectivity(min_element_size)

    # スーパーピクセルセグメンテーションの境界を取得
    contour_mask = slc.getLabelContourMask(False)
    image[0 < contour_mask] = (0, 0, 255)
    cv2.imshow('LSC result', image)
    cv2.waitKey(0)

    #test

if __name__ == '__main__':
    main()