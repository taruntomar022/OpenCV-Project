import numpy as np
import matplotlib as plt
import cv2

# img = cv2.imread('demo.jpg',1)
#
# img[550,550] = [0,255,0]
# img[100:150, 100:150] = [255,255,255]

# cv2.line(img,(0,0),(350,350),(255,255,255),15,)
# cv2.rectangle(img,(50,50),(450,450),(255,0,0),5)
# cv2.circle(img,(350,350),20,(0,0,255),2)
# font = cv2.FONT_HERSHEY_COMPLEX
# cv2.putText(img,'OpenCV',(500,500),font,1,(255,255,0),2,cv2.LINE_8)

# cv2.imshow('image',img)
# cv2.waitKey(0)
# cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.ovi',fourcc,20.0,(640,480))


# img1 = cv2.imread('3D-Matplotlib.png')
# img2 = cv2.imread('mainsvmimage.png')
#
#
# weighted = cv2.addWeighted(img1,0.7,img2,0.3,0)


# add = img1 + img2
# add = cv2.add(img1,img2)
# cv2.imshow('weighted',weighted)

# while True:
#     ret,frame = cap.read()
#     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     out.write(frame)
#     cv2.imshow('frame',frame)
#     cv2.imshow('gray',gray)
#     if cv2.waitKey(1) & 0xFF==ord('q'):
#         break
# cap.release()
# out.release()
#........................color filtering...............//
# cap = cv2.VideoCapture(0)
# while True:
#     _, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     lower_white = np.array([20, 0, 0])
#     upper_white = np.array([180, 180, 180])
#     mask = cv2.inRange(hsv, lower_white, upper_white)
#     res = cv2.bitwise_and(frame, frame, mask=mask)
#     cv2.imshow('frame', frame)
#     cv2.imshow('mask', mask)
#     cv2.imshow('res', res)
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
#
# # cv2.waitKey(0)
# cv2.destroyAllWindows()
# cap.release()
#.......................blurring  smoothing.................................//

cap = cv2.VideoCapture(0)

cap.set(3,50)
cap.set(4,50)
while True:
    _,frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([150, 90, 0])
    upper_white = np.array([180, 255, 180])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()

