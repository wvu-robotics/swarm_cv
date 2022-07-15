import cv2
import numpy as np

cap = cv2.VideoCapture('C:\\Users\\katel\\OneDrive\\Desktop\\Flock_Trim_Begin.mp4')
positions = []
i=0
frame = []
j=0
data = []
k=0
#black_min = np.array([0,0,0])
#black_max = np.array([60,60,60])

while True:
    (ret,imgPrev) = cap.read()
    if imgPrev is None:
        break
    background = cv2.cvtColor(imgPrev, cv2.COLOR_BGR2GRAY)
    #background = cv2.inRange(background,black_min,black_max)
    (ret,img) = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #hsv = cv2.inRange(hsv,black_min,black_max)
    remove = cv2.absdiff(background,hsv)
    threshold = cv2.dilate(remove,None,iterations=3)
    #cv2.imshow("Remove",threshold)
    threshold = cv2.threshold(threshold,20,255,cv2.THRESH_BINARY)[1]
    threshold = cv2.GaussianBlur(threshold,(7,7),cv2.BORDER_DEFAULT)
    #cv2.imshow("Threshold",threshold)
    contours = threshold.copy()
    outlines, hierarchy = cv2.findContours(contours,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    positions = []
    for c in outlines:
        if cv2.contourArea(c) < 1000:
            continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(img,(cX, cY),7,(179,100,255),-1)
        positions.append([cX,cY])
        i = i+1
        #cv2.imshow("Camera",img)
        #cv2.imshow("Threshold",threshold)
        #cv2.imshow("Subtraction",remove)
        #cv2.imshow("Countour",contours)
    positions=np.array(positions)
    if len(positions)>0:data.append(positions)

data=np.array(data)
print(data)
cap.release()
cv2.destroyAllWindows()