# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 23:13:23 2021

@author: ricar
"""

import cv2
import os
import numpy as np
from PIL import Image
from time import time
import imutils

tiempo_inicial = time()

personName = 'Ricardo'
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
       
        cv2.imwrite("C:/Users/ricar/Desktop/OPeracion/Data/Ricardo/" + str(Id) + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
       
        sampleNum = sampleNum +1
        cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif sampleNum > 120:
        break


detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

imagepath = "C:/Users/ricar/Desktop/Operacion/Data/Ricardo/"

def get_images_and_labels(path):
    image_paths = [os.path.join(imagepath, f) for f in os.listdir(imagepath)]
    face_samples = []
    ids = []
    for image_path in image_paths:
           
        image = Image.open(image_path).convert('L')
        
        image = cv2.resize(auxFrame,(150,150),interpolation=cv2.INTER_CUBIC)
               
        image = np.array(image, 'uint8')
                 
        if os.path.split(image_path)[-1].split(".")[-1] != 'jpg':
        #if os.path.split(image)[-1].split(".")[-1] != 'jpg':
            continue
        image_id = int(os.path.split(image_path)[-1].split(".")[1])
        #faces = detector.detectMultiScale(image_np)
        faces = detector.detectMultiScale(image)

        for (x, y, w, h) in faces:
            #face_samples.append(image_np[y:y + h, x:x + w])
            face_samples.append(image[y:y + h, x:x + w])
            ids.append(image_id)
            #ids.append(image)
    return face_samples, ids

faces, Ids = get_images_and_labels('dataSet')
recognizer.train(faces, np.array(Ids))


recognizer.write('modeloLBPH.xml')
print("modelo almacenado...")

tiempo_final = time()
tiempo_ejecucion = tiempo_final - tiempo_inicial
print("Tiempo de ejecucion (s): ", tiempo_ejecucion)
cap.release()
cv2.destroyAllWindows()
#HASTA AQUI OK


tiempo_inicial = time()


#imagesPath = "C:/Users/ricar/Desktop/Ricardo Rojas/Data/Datos1"
imagesPath = "C:/Users/ricar/Desktop/Operacion/Data/Ricardo"
imagesPathlist = os.listdir(imagesPath)
print('imagesPathlist=', imagesPathlist)

if not os.path.exists('Rostros encontrados_10'):
    print('Carpeta creada: Rostros encontrados_10')
    os.makedirs('Rostros encontrados_10')
    
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 0

for imageName in imagesPathlist:
    print('imagesName=',imageName)
    image = cv2.imread(imagesPath+'/'+imageName)
    #imageAux = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageAux = gray.copy()
    faces = faceClassif.detectMultiScale(gray, 1.1, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(128,0,255),2)
    cv2.rectangle(gray,(10,5),(450,25),(255,255,355),-1)
    #cv2.putText(gray,'Presione a, para almacenar los rostros encontrados',(10,20),2,0.5,(128,0,255),1,cv2.LINE_AA)
    #cv2.imshow('image',gray)
    #k = cv2.waitKey(0)
    #if k == ord('a'):
    for(x,y,w,h) in faces:
            rostro = imageAux[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('Rostros encontrados_10/rostro_{}.jpg'.format(count),rostro)
            count = count + 1
    #elif k == 27:
    #    break
            
        #cv2.imshow('image', image)
    
    #cv2.imshow('image',image)
    #cv2.waitKey(0)
    
faces, Ids = get_images_and_labels('dataSet')
recognizer.train(faces, np.array(Ids))


recognizer.write('modeloLBPH.xml')
print("modelo almacenado...")
    
tiempo_final = time()
tiempo_ejecucion = tiempo_final - tiempo_inicial
print("Tiempo de ejecucion (s): ", tiempo_ejecucion)
cv2.destroyAllWindows()


#HASTA AQUI OK 2

dataPath = dataPath = 'C:/Users/ricar/Desktop/Operacion/Data'
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('leyendo las imagenes')
    
    for fileName in os.listdir(personPath):
        print ('Carpeta: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
        image = cv2.imread(personPath+'/'+fileName,0)
    label = label + 1

#print('labels= ', labels)
#print ('Numero de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
#print ('Numero de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))
#print ('Numero de etiquetas 2: ',np.count_nonzero(np.array(labels)==2))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Entrenando el modelo
print ("Entrenando...")
face_recognizer.train(facesData, np.array(labels))
#Almacenando el modelo obtenido
face_recognizer.write('modeloLBPHFace1108.xml')
print("Modelo almacenado...")
tiempo_final = time()
tiempo_ejecucion = tiempo_final - tiempo_inicial
print("Tiempo de ejecucion (s): ", tiempo_ejecucion)

cv2.waitKey(0)
cv2.destroyAllWindows()