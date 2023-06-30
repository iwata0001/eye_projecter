import cv2
import numpy as np

def main():
    image = cv2.imread('data_eyes/001.png')
    image = cv2.resize(image, (640, 480))
    #test
    #test2

if __name__ == '__main__':
    main()