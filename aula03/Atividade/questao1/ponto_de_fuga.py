#!/usr/bin/python
# -*- coding: utf-8 -*-


import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import sys
    


def region_of_interest(img, regiao):
    height = img.shape[0]
    width = img.shape[1]

    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    channel_count = img.shape[2]
    # Create a match color with the same color channel counts.
    match_mask_color = (255,) * channel_count

    corte_pista = None
      
    if regiao == 'esquerda':
        corte_pista = [(0, height), (0, height/2), (width/2, height/2),(width/2, height), (width, height),]

    elif regiao == 'direita':
        corte_pista = [(width, height), (width, height/2), (width/2, height/2),(width/2, height), (0,height),]


    cv2.fillPoly(mask, np.array([corte_pista], np.int32), match_mask_color)
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)


    return masked_image


cap = cv2.VideoCapture('line_following.mp4')


lower = 0
upper = 1


while(True):
    kernel = np.ones((3, 3),np.uint8)

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Cortando a imagem para a análise somente das regiões de interesse
    pista_esquerda = region_of_interest(frame, 'esquerda',)
    pista_direita = region_of_interest(frame, 'direita',)
    
    # Definindo uma másquina da cor branca (cor da pista)
    hsv_1, hsv_2 = np.array([0,0,240], dtype=np.uint8), np.array([255,15,255], dtype=np.uint8)
    
    hsv_esquerda = cv2.cvtColor(pista_esquerda , cv2.COLOR_BGR2HSV) 
    hsv_direita = cv2.cvtColor(pista_direita , cv2.COLOR_BGR2HSV)   

    color_mask_esquerda = cv2.inRange(hsv_esquerda, hsv_1, hsv_2)
    color_mask_direita = cv2.inRange(hsv_direita, hsv_1, hsv_2)
    

    # IDENTIFICAÇÃO DAS RETAS:
    x1e, y1e, x2e, y2e = None, None, None, None
    x1d, y1d, x2d, y2d = None, None, None, None
    md, me = None, None

    # Identificando as linhas na parte esquerda da pista
    lines_esquerda = cv2.HoughLinesP(color_mask_esquerda, 1, np.pi/180, 30, maxLineGap=200)
    # draw Hough lines
    for line in lines_esquerda:
        x1e, y1e, x2e, y2e = line[0]
        me = ((y1e - y2e)*1.0)/(x1e - x2e)
        cv2.line(frame, (x1e, y1e), (x2e, y2e), (0, 255, 0), 3)


    # Identificando as linhas na parte direita da pista
    lines_direita = cv2.HoughLinesP(color_mask_direita, 1, np.pi/180, 30, maxLineGap=200)
    # draw Hough lines
    for line in lines_direita:
        x1d, y1d, x2d, y2d = line[0]
        md = ((y1d - y2d)*1.0)/((x1d - x2d))
        cv2.line(frame, (x1d, y1d), (x2d, y2d), (255, 0, 0), 3)


    # Equação da reta esquerda:
    # y - y1e = me(x - x1e) 
    # y = me*x + y1e - me*x1e 

    # Equação da reta direita:
    # y - y1d = md(x - x1d) 
    # y = md*x + y1d - md*x1d 

    # Intersecção:
    # me*x + y1e - me*x1e  = md*x + y1d - md*x1d
    
    x = ((y1d - y1e + me*x1e - md*x1d)*1.0)/(me - md)
    y =  me*x + y1e - me*x1e

    print('----------------------')
    print('x: {}'.format(x))
    print('y: {}'.format(y))
    print('----------------------')

    # Ponto de Fuga
    cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), 6)


    cv2.imshow("Ponto de Fuga", frame)

    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows() 