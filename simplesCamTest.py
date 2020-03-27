
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # setando altura
cap.set(4, 480)  # setando largura

while (True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # virar camera verticalmente
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # pressionar tecla 'ESC' pra parar
        break

cap.release()
cv2.destroyAllWindows()
