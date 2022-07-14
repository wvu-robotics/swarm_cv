import cv2
import numpy as np
import time

cap = cv2.VideoCapture('C:\\Users\\katel\\OneDrive\\Desktop\\Robot_Ex.mp4')
black_min = np.array([0,0,0])
black_max = np.array([60,60,60])

while True:
    (ret,imgPrev) = cap.read()
    background = cv2.cvtColor(imgPrev, cv2.COLOR_BGR2HSV)
    background = cv2.inRange(background,black_min,black_max)
    (ret,img) = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.inRange(hsv,black_min,black_max)
    remove = cv2.absdiff(background,hsv)
    threshold = cv2.threshold(remove,20,255,cv2.THRESH_BINARY)[1]
    threshold = cv2.GaussianBlur(threshold,(7,7),cv2.BORDER_DEFAULT)
    contours = threshold.copy()
    outlines, hierarchy = cv2.findContours(contours,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in outlines:
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("Camera",img)
        #cv2.imshow("Threshold",threshold)
        #cv2.imshow("Subtraction",remove)
        cv2.imshow("Countour",contours)
        key = cv2.waitKey(24) & 0xFF
        time.sleep(0.05)
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()