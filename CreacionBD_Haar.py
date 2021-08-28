# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:30:29 2021

@author: ricar
"""


import cv2
import os
import numpy as np
from PIL import Image
from time import time
import imutils

tiempo_inicial = time()

personName = 'Ricardo2'
dataPath = 'C:/Users/ricar/Desktop/Operacion/Data'
personPath = dataPath + '/'+personName
print(personPath)
if not os.path.exists(personPath):
    print('Carpeta Creada: ', personPath)
    os.makedirs(personPath)


detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
sampleNum = 0
Id = input('Enter your Id: ')
while True:
    ret, img = cap.read()
    if ret == False:break
    img = imutils.resize(img, width=640)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    
    faces = detector.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y),(x + w, y + h), (255,0,0),2)
   
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(auxFrame,(150,150),interpolation=cv2.INTER_CUBIC)
       
        cv2.imwrite("C:/Users/ricar/Desktop/OPeracion/Data/Ricardo2/" + str(Id) + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
       
        sampleNum = sampleNum +1
        cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif sampleNum > 120:
        break
tiempo_final = time()
tiempo_ejecucion = tiempo_final - tiempo_inicial
print("Tiempo de ejecucion (s): ", tiempo_ejecucion)
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()