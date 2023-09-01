import cv2
import numpy as np
import matplotlib.pyplot as plt
# test

def main():
    mask = cv2.imread('temp_img/eyeMask.png')
    mask = mask * 255

    eye = cv2.imread('output/_wv_output.png')
    k = 0.97

    resizeMask = cv2.resize(mask, None, fx = k, fy = k)
    resizeEye = cv2.resize(eye, None, fx = k, fy = k)

    size = resizeMask.shape
    print(size)

    back = cv2.imread('preview/1_noLEye.png')

    maskBig = np.zeros((300,300,3), np.uint8)
    eyeBig = np.zeros((300,300,3), np.uint8)

    maskBig[0:size[0], 0:size[1]] = resizeMask
    eyeBig[0:size[0], 0:size[1]] = resizeEye
    center = (185, 144)

    result = cv2.seamlessClone(eyeBig, back, maskBig, center, cv2.NORMAL_CLONE)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    plt.imshow(result)
    plt.show()

if __name__ == '__main__':
    main()