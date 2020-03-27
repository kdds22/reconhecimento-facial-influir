# precisa instalar o: opencv-contrib-python (cv2.face) e o pillow

import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:/Users/Influir/Documents/projects/FaceReco/trainer/trainer.yml')
cascadePath = "C:/Users/Influir/Documents/projects/FaceReco/Cascades/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX # text-font

id = 0 # inicializando o match

# enum: ID - NAME
names = ['Pessoa1', 'Pessoa2']

cam = cv2.VideoCapture(0) # abrindo camera

cam.set(3, 640)  # largura
cam.set(4, 480)  # altura

# Tamanho minimo da tela de reconhecimento
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Flip vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w]) # Pega o ID das faces e a similaridade

        if 100 >= confidence > 30: # Maior que 0 é pq tem alguma similaridade
            id = names[id] # pega o nome
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "Desconhecido"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff  # 'ESC' pra parar
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
