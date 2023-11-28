import cv2
import numpy as np
import matplotlib.pyplot as plt
# test2

def main():
    path_w = 'test_result/result.txt'

    s = 'New file\nkaigyou\nslash\\\ndouble\\\\\n'
    v = 'another\n'

    with open(path_w, mode='w') as f:
        f.write(s)
        for i in range(3):
            f.write(v)

if __name__ == '__main__':
    main()