import cv2
import numpy as np
from PIL import Image
import os
# caminho do dataset dos rostos
path = 'C:/Users/Influir/Documents/projects/FaceReco/dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('C:/Users/Influir/Documents/projects/FaceReco/Cascades/haarcascades/haarcascade_frontalface_default.xml')


def getImagesAndLabels(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    names = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # escala de cinza
        img_numpy = np.array(PIL_img, 'uint8')
        name = str(os.path.split(imagePath)[-1].split("_")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            names.append(name)
    return faceSamples, names


print("\n ---> Rostos em treinamento <---\n")

faces, names = getImagesAndLabels(path)

labels = []
for i in range(len(np.unique(names))): # pra cada usuario...
    for a in range(len(names)): # verificar quantos existe no array
        if names[a] == np.unique(names)[i]:
            labels.append(i) # acrescentando o ID do rosto
            print(i)
            print(names[a])
    pass
print(names, labels)
names = labels # setando os indeces aos nomes

recognizer.train(faces, np.array(names))

recognizer.write('C:/Users/Influir/Documents/projects/FaceReco/trainer/trainer.yml')

print("\n ---> {0} faces trainadas.".format(len(np.unique(names))))
