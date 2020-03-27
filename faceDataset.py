import cv2
import os

face_detect = cv2.CascadeClassifier('C:/Users/Influir/Documents/projects/FaceReco/Cascades/haarcascades/haarcascade_frontalface_default.xml')
face_name = input('\n Informe o Nome do Usuario e pressione Enter: ')
print('\nInicializando Captura da Imagem\nOlhe para a CAMERA e ESPERE')

cam = cv2.VideoCapture(0);
cam.set(3, 640)
cam.set(4, 480)

count = 0

while (True):
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2) # imagem, ponto_inicial, ponto_final, cor BGR, espessura)
        count += 1
        cv2.imwrite('C:/Users/Influir/Documents/projects/FaceReco/dataset/User_'+str(face_name)+'_'+str(count)+'.jpg', gray[y:y+h, x:x+w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # esc
    if k == 27:
        break
    elif count >= 100: # quantidade de treino por rosto
        break

print('\nExiting...')

cam.release()
cv2.destroyAllWindows()

