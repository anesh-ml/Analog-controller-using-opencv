import cv2
import numpy as np
import math
from scipy.spatial import distance
import imutils
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pyglet
def Zoom(cv2Object, zoomSize):
    # Resizes the image/video frame to the specified amount of "zoomSize".
    # A zoomSize of "2", for example, will double the canvas size
    cv2Object = imutils.resize(cv2Object, width=(int(round((1-zoomSize) * img.shape[1]))))
    # center is simply half of the height & width (y/2,x/2)
    center = (cv2Object.shape[0]/2,cv2Object.shape[1]/2)
    # cropScale represents the top left corner of the cropped frame (y/x)
    cropScale = (center[0]/1), (center[1]/1)
    # The image/video frame is cropped to the center with a size of the original picture
    # image[y1:y2,x1:x2] is used to iterate and grab a portion of an image
    # (y1,x1) is the top left corner and (y2,x1) is the bottom right corner of new cropped frame.
    cv2Object = cv2Object[int(cropScale[0]):(round(int(center[0] + cropScale[0]))), int(cropScale[1]):(round(int(center[1] + cropScale[1])))]
    return cv2Object
Ratio=[]
mean=14435.271941063422
std=1501.9989122573215
j=0
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    # read image
    ret, img = cap.read()
    width = img.shape[1] 
    height = img.shape[0]
    # get hand data from the rectangle sub window on the screen
    cv2.rectangle(img, (350,350), (100,100), (0,255,0),0)
    crop_img = img[100:350, 100:350]

    # convert to grayscale
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # applying gaussian blur
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)

    # thresholdin: Otsu's Binarization method
    _, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # show thresholded image
   # cv2.imshow('Thresholded', thresh1)

    # check OpenCV version to avoid unpacking error
    (version, _, _) = cv2.__version__.split('.')

    if version == '3':
        image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    elif version == '2':
        contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
               cv2.CHAIN_APPROX_NONE)
    
    # find contour with max area
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    area_cnt=cv2.contourArea(cnt)
    
    area_ratio=(area_cnt-mean)/(std)
    ratio=1/(1+math.exp(area_ratio))
    #ratio_1=(ratio-0.3)/(0.9-0.3)
    Ratio.append(ratio)       
    if(len(Ratio)>2):
        Ratio[:]=[]
    ratio_1=(ratio-(0.0033)/(0.99-0.0033))*(-65-(-1))-1
    img=Zoom(img,ratio)
    
    #print(j)
    # create bounding rectangle around the contour (can skip below two lines)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)    
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
# show appropriate images in windows
    cv2.imshow('Gesture', img)
    #cv2.imshow('Contours', all_img)
    #cv2.imshow('zoom', img1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()