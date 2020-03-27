import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('C:/Users/Influir/Documents/projects/FaceReco/Cascades/haarcascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('C:/Users/Influir/Documents/projects/FaceReco/Cascades/haarcascades/haarcascade_eye.xml')
eyeGlassCascade = cv2.CascadeClassifier('C:/Users/Influir/Documents/projects/FaceReco/Cascades/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
smileCascade = cv2.CascadeClassifier('C:/Users/Influir/Documents/projects/FaceReco/Cascades/haarcascades/haarcascade_smile.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
cap.set(3, 640) # set Width
cap.set(4, 480) # set Height
while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.putText(img, 'Rosto', (x, y), font, 2, (255, 0, 0), 5)

        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=250,
            minSize=(20, 5),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sh, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            cv2.putText(img, 'Boca', (x + sx, y + sy), 1, 1, (0, 0, 255), 1)

        #'''
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.25,
            minNeighbors=10,
            minSize=(5, 5),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.putText(img, 'Olho', (x + ex, y + ey), 1, 1, (0, 255, 0), 1)
        '''

        eyesGlass = eyeGlassCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.30,
            minNeighbors=20,
            minSize=(20, 5),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (egx, egy, egw, egh) in eyesGlass:
            cv2.rectangle(roi_color, (egx, egy), (egx + egw, egy + egh), (0, 150, 150), 2)
            cv2.putText(img, 'Oculos', (x + egx, y + egy), 1, 1, (0, 150, 150), 1)'''

    cv2.putText(img, 'Numero de Rostos : ' + str(len(faces)), (40, 40), font, 1, (255, 0, 0), 2)


    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()

