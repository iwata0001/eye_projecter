import cv2
import numpy as np

def main():
    a = np.array([0,1,2,3,4,5,6,7,8,9])
    b = np.array([10,11,12,13])

    a[[0,2,4,6]] = b

    print(a)

if __name__ == '__main__':
    main()