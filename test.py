import cv2
import numpy as np

def main():
    a = np.arange(100).reshape(10,10)
    print(a)
    b=np.delete(a,[2,4,6],1)
    print(b)

    c=np.array([[[1,2]],[[3,4]],[[5,6]],[[7,8]]])
    d=np.array([[[1,1]],[[1,1]],[[1,1]],[[1,1]]])
    e=np.array([[1,1]])
    print(c-e)

if __name__ == '__main__':
    main()