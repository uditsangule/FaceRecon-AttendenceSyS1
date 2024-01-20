import cv2
import numpy as np



def loadimage(imagepath , mode = None):
    cap = cv2.VideoCapture(0 if imagepath is None else imagepath)
    return cv2.imread(filename=imagepath)

def showimage(image , windowname='faceout' , waitkey=0):
    cv2.namedWindow(windowname)
    cv2.imshow(windowname , image)
    cv2.waitKey(waitkey)
    cv2.destroyWindow(windowname)
    return

