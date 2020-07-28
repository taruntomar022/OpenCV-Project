import cv2
import numpy as np
import pickle


face_cas = cv2.CascadeClassifier('Cascade/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels ={}
with open("lable.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

# cap.set(3,50)
# cap.set(4,50)

while True:
    _,frame = cap.read()

    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(grey,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        # print(x,y,w,h)
        roi_grey = grey[y:y+h,x:x+w]

        id_,conf = recognizer.predict(roi_grey)
        if conf >=45 and conf<=85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_COMPLEX
            color=(255,255,255)
            stroke=2
            name= labels[id_]
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
            
        img_item = "my_image.png"
        cv2.imwrite(img_item,grey)

        color = (255,0,0)
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()