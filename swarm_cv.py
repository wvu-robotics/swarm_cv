import cv2
import time

cap = cv2.VideoCapture('C:\\Users\\katel\\OneDrive\\Desktop\\Rink_Swarm.mp4')

while True:
    (ret,imgPrev) = cap.read()
    background = cv2.cvtColor(imgPrev, cv2.COLOR_BGR2GRAY)
    (ret,img) = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    remove = cv2.absdiff(background,gray)
    remove = cv2.GaussianBlur(remove,(3,3),cv2.BORDER_DEFAULT)
    threshold = cv2.threshold(remove,20,255,cv2.THRESH_BINARY)[1]
    threshold = cv2.dilate(threshold,None,iterations = 2)
    contours = threshold.copy()
    outlines, hierarchy = cv2.findContours(contours,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in outlines:
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("Camera",img)
        cv2.imshow("Threshold",threshold)
        cv2.imshow("Subtraction",remove)
        cv2.imshow("Countour",contours)
        key = cv2.waitKey(24) & 0xFF
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()