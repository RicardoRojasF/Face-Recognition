# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 21:38:58 2021

@author: ricar
"""

import cv2
import os
import numpy as np
from PIL import Image
from time import time
import imutils

tiempo_inicial = time()

personName = 'Ampi'
dataPath = 'C:/Users/ricar/Desktop/Operacion/Data'
personPath = dataPath + '/'+personName
print(personPath)
if not os.path.exists(personPath):
    print('Carpeta Creada: ', personPath)
    os.makedirs(personPath)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

sampleNum = 0
Id = input('Enter your Id: ')
while True:
    ret, img = cap.read()
    if ret == False:break
    img = imutils.resize(img, width=640)
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    
    faces = detector.detectMultiScale(img, 1.3, 5)
    rostro = cv2.resize(gray,(150,150),interpolation=cv2.INTER_CUBIC)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y),(x + w, y + h), (255,0,0),2)
   
        rostro = gray[y:y+h,x:x+w]
        rostro = cv2.resize(gray,(150,150),interpolation=cv2.INTER_CUBIC)
       
        cv2.imwrite("C:/Users/ricar/Desktop/Operacion/Data/Ampi/" + str(Id) + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
       
        sampleNum = sampleNum +1
        cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif sampleNum > 110:
        break

cap.release()
cv2.destroyAllWindows()

#imagesPath = "C:/Users/ricar/Desktop/Ricardo Rojas/Data/Datos1"
imagesPath = "C:/Users/ricar/Desktop/Operacion/Data/Ampi"
imagesPathlist = os.listdir(imagesPath)
print('imagesPathlist=', imagesPathlist)

if not os.path.exists('Rostros encontrados'):
    print('Carpeta creada: Rostros encontrados')
    os.makedirs('Rostros encontrados')
    
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 0

for imageName in imagesPathlist:
    print('imagesName=',imageName)
    image = cv2.imread(imagesPath+'/'+imageName)
    imageAux = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageAux = gray.copy()
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(128,0,255),2)
        cv2.rectangle(gray,(10,5),(450,25),(255,255,355),-1)
        cv2.putText(gray,'Presione a, para almacenar los rostros encontrados',(10,20),2,0.5,(128,0,255),1,cv2.LINE_AA)
        cv2.imshow('image',gray)
    k = cv2.waitKey(0)
    if k == ord('a'):
        for(x,y,w,h) in faces:
            rostro = imageAux[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('Rostros encontrados/rostro_{}.jpg'.format(count),rostro)
        count = count + 1
    elif k == 27:
        break
            
        #cv2.imshow('image', image)
    
    #cv2.imshow('image',image)
    #cv2.waitKey(0)
    
recognizer.write('modeloLBPH.xml')
print("modelo almacenado...")
tiempo_final = time()
tiempo_ejecucion = tiempo_final - tiempo_inicial
print("Tiempo de ejecucion (s): ", tiempo_ejecucion)
#cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()
