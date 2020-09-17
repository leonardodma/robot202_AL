#!/usr/bin/python
# -*- coding: utf-8 -*-


# Referências:
# https://www.geeksforgeeks.org/circle-detection-using-opencv-python/   
# https://www.geeksforgeeks.org/python-opencv-cv2-line-method/


import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import sys
import auxiliar as aux 


def encontra_circulo(img, codigo_cor):
    # MAGENTA
    hsv_1, hsv_2 = aux.ranges(codigo_cor)

    # convert the image to grayscale, blur it, and detect edges
    hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    
    color_mask = cv2.inRange(hsv, hsv_1, hsv_2)
   
    segmentado = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((10, 10)))
    
    segmentado = cv2.adaptiveThreshold(segmentado,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                 cv2.THRESH_BINARY,11,3.5)

    kernel = np.ones((3, 3),np.uint8)
	
    segmentado = cv2.erode(segmentado,kernel,iterations = 1)
    
    circles=cv2.HoughCircles(segmentado, cv2.HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=5,maxRadius=100)
    
    return circles


if len(sys.argv) > 1:
    arg = sys.argv[1]
    try:
        input_source=int(arg) # se for um device
    except:
        input_source=str(arg) # se for nome de arquivo
else:   
    input_source = 0

cap = cv2.VideoCapture(input_source)

# Parameters to use when opening the webcam.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lower = 0
upper = 1


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    circles_magenta = encontra_circulo(frame, '#FF00FF')
    circles_ciano = encontra_circulo(frame, '#5dbcce')

    x_m, y_m, r_m = None, None, None
    x_c, y_c, r_c = None, None, None


    if circles_magenta is not None:
        circles_magenta = np.uint16(np.around(circles_magenta)) 
    
        # Desenha círculos da cor Magenta 
        for pt in circles_magenta[0, :]: 
            x_m, y_m, r_m = pt[0], pt[1], pt[2] 
    
            # Draw the circunference of the circle. 
            cv2.circle(frame, (x_m, y_m), r_m, (0, 255, 255), 2) 
    
            # Draw a small circle (of radius 1) to show the center. 
            cv2.circle(frame, (x_m, y_m), 1, (0, 255, 255), 3)


    if circles_ciano is not None: 
        circles_ciano= np.uint16(np.around(circles_ciano)) 
        # Desenha círculos da cor ciano 
        for pt in circles_ciano[0, :]: 
            x_c, y_c, r_c = pt[0], pt[1], pt[2] 
    
            # Draw the circunference of the circle. 
            cv2.circle(frame, (x_c, y_c), r_c, (0, 255, 255), 2) 
    
            # Draw a small circle (of radius 1) to show the center. 
            cv2.circle(frame, (x_c, y_c), 1, (0, 255, 255), 3)


    centro_ciano = tuple([x_c, y_c])
    centro_magenta = tuple([x_m, y_m])

    print(centro_ciano)

    angle = None
    
    if centro_ciano[0] != None or centro_magenta[0] != None:
        try:
            line = cv2.line(frame, centro_ciano, centro_magenta, (255, 0, 0), 6)
            angle = math.atan2(y_m - y_c, x_m - x_c)
            angle = angle * (180/math.pi)
            print(angle)
        except:
            pass

    if angle != None:
        cv2.putText(frame, "Angulo: %.2f graus" % (angle),(frame.shape[1] - 600, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)


    cv2.imshow("Detected Circle", frame)
    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()