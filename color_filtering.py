import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while True:
    _,frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    lower_white = np.array([0,0,0])
    upper_white = np.array([255,255,255])
    cv2.imshow('frame',frame)
    mask = cv2.inRange(hsv,lower_white,upper_white)
    res=cv2.bitwise_and(frame,frame,mask=mask)
    
    cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()