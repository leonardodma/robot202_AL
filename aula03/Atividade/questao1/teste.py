#!/usr/bin/python
# -*- coding: utf-8 -*-


import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import sys
    

cap = cv2.VideoCapture('line_following.mp4')


lower = 0
upper = 1


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    hsv_1, hsv_2 = np.array([0,0,200], dtype=np.uint8), np.array([0,0,255], dtype=np.uint8)

    # convert the image to grayscale, blur it, and detect edges
    hsv = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)    
    
    color_mask = cv2.inRange(hsv, hsv_1, hsv_2)

    segmentado = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((10, 10)))
    
    segmentado = cv2.adaptiveThreshold(segmentado,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,3.5)

    kernel = np.ones((3, 3),np.uint8)
	
    segmentado = cv2.erode(segmentado,kernel,iterations = 1)
    
    lines = cv2.HoughLinesP(segmentado,rho=6,theta=np.pi / 60,threshold=160,
                            lines=np.array([]),minLineLength=80,maxLineGap=25)
    

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(frame, (x1, y1), (x2, y2), [0, 0, 255], 3)


    cv2.imshow("Ponto de Fuga", frame)

    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows() 