import cv2
import numpy as np

cap = cv2.VideoCapture('C:\\Users\\katel\\OneDrive\\Desktop\\Flock_Trim_Begin_Short.mp4')
positions = []
i=0
data = []

while True:
    (ret,imgPrev) = cap.read()
    if imgPrev is None:
        break
    background = cv2.cvtColor(imgPrev, cv2.COLOR_BGR2GRAY)
    (ret,img) = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    remove = cv2.absdiff(background,gray)
    threshold = cv2.dilate(remove,None,iterations=4)
    threshold = cv2.threshold(threshold,10,255,cv2.THRESH_BINARY)[1]
    threshold = cv2.GaussianBlur(threshold,(7,7),cv2.BORDER_DEFAULT)
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
        #time.sleep(0.05)
        positions.append([cX,cY])
        i = i+1
        cv2.imshow("Camera",img)
        #cv2.imshow("Threshold",threshold)
        #cv2.imshow("Subtraction",remove)
        #cv2.imshow("Countour",contours)
    
    key = cv2.waitKey(24) & 0xFF
    if key == ord("q"):
        break

    positions=np.array(positions)
    if len(positions) > 0:data.append(positions)
data=np.array(data)
print(data)
outfile = 'Swarm_Positions'
np.savez(outfile,data=data)
cap.release()
cv2.destroyAllWindows()