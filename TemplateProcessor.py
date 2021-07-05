import cv2 as cv
import numpy as np
from sklearn import cluster
import time


from main import *

if __name__ == '__main__':
    img, _, _ = img_read("a.png")
    img = cv.Canny(img, 300, 500)
    cv.imwrite("b.png", img)

