import numpy as np
import cv2 as cv
import matplotlib.pyplot as plot

cap = cv.VideoCapture(0)

while(1):
    # get a frame
    ret, img = cap.read()
    # show a frame
    cv.imshow("capture", img)
    if cv.waitKey(1) & 0xFF == ord(' '):
        break

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
# img = cv.imread('zzz.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    i=1
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    # roi = img[y:y + height, x:x + width]
        roi = roi_color[ey:ey+eh, ex:ex+ew ]       #按照脸这张图的为大坐标的基础上截的 因为眼睛一点在脸上～
        cv.imwrite(str(i)+"roi.jpg", roi)           #两个眼睛所以设置不同的i保存的图片名字不一样
        i=i+1

cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
