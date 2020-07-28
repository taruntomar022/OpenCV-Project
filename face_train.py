import os
import cv2
from PIL import Image
import numpy as np
import pickle

face_cas = cv2.CascadeClassifier('Cascade/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR,"images")

current_id =0
label_ids={}
y_labels =[]
x_train = []

for root,dir,files in os.walk(IMAGE_DIR):
    for file in files:
        if file.endswith("png") or file.endswith("jpeg") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            # print(label,path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id+=1
            id_ = label_ids[label]
            # print(label_ids)

            pil_img = Image.open(path).convert("L")
            image_arr = np.array(pil_img,"uint8")
            # print(image_arr)
            faces = face_cas.detectMultiScale(image_arr,scaleFactor=1.5,minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_arr[y:y + h, x:x + w]
                x_train.append(roi)
                y_labels.append(id_)
# print(x_train)
# print(y_labels)

with open("lable.pickle",'wb') as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainer.yml")