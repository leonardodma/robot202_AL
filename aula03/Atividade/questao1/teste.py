#!/usr/bin/python
# -*- coding: utf-8 -*-


import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import sys
    


def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    channel_count = img.shape[2]
    # Create a match color with the same color channel counts.
    match_mask_color = (255,) * channel_count
      
    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


cap = cv2.VideoCapture('line_following.mp4')


lower = 0
upper = 1


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    height = frame.shape[0]
    width = frame.shape[1]
    
    region_of_interest_vertices = [(0, height), (0, height/2), (width, height/2), (width,height)]
    
    cropped_image = region_of_interest(frame,np.array([region_of_interest_vertices], np.int32),)
    
    hsv_1, hsv_2 = np.array([0,0,240], dtype=np.uint8), np.array([255,15,255], dtype=np.uint8)

    # convert the image to grayscale, blur it, and detect edges
    hsv = cv2.cvtColor(cropped_image , cv2.COLOR_BGR2HSV)    
    
    color_mask = cv2.inRange(hsv, hsv_1, hsv_2)

    segmentado = cv2.adaptiveThreshold(color_mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,3.5)

    kernel = np.ones((3, 3),np.uint8)
	
    segmentado = cv2.erode(segmentado,kernel,iterations = 1)
    
    


    cv2.imshow("Ponto de Fuga", segmentado)

    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows() 