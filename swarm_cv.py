import cv2
import numpy as np
import math

cap = cv2.VideoCapture('CV_Sample_Videos/flock_trim.mp4')
positions = []
i=0
j=0
data = []
parse = []
xVal = []
yVal = []

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
        xVal.append(cX)
        yVal.append(cY)
        cv2.circle(img,(cX, cY),7,(179,100,255),-1)
        #time.sleep(0.05)
        positions.append([cX,cY])
        i = i+1
        # cv2.imshow("Camera",img)
        # cv2.imshow("Threshold",threshold)
        # cv2.imshow("Subtraction",remove)
        # cv2.imshow("Countour",contours)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    positions=np.array(positions)
    if len(positions) > 0:data.append(positions)
data=np.array(data)
enclosureX = max(xVal)
enclosureY = max(yVal)
print(data)
if xVal >= yVal:
    enclosure = enclosureX + enclosureX*.10
else:
    enclosure = enclosureY + enclosureY*.10
#dataNew = []
#while j < len(data)-1:
    #k = 0
    #dataChunk = []
    #while k < 5:
        #distance = []
        #distance.append(math.dist([data[j][k][0],data[j][k][1]],[data[j+1][0][0],data[j+1][0][1]]))
        #distance.append(math.dist([data[j][k][0],data[j][k][1]],[data[j+1][1][0],data[j+1][1][1]]))
        #distance.append(math.dist([data[j][k][0],data[j][k][1]],[data[j+1][2][0],data[j+1][2][1]]))
        #distance.append(math.dist([data[j][k][0],data[j][k][1]],[data[j+1][3][0],data[j+1][3][1]]))
        #distance.append(math.dist([data[j][k][0],data[j][k][1]],[data[j+1][4][0],data[j+1][4][1]]))
        #mini = min(distance)
        #agent = distance.index(mini)
        #dataChunk.append([data[j][agent][0],data[j][agent][1]])
        #if k == 4:
            #dataNew.append(dataChunk)
        #k = k+1
    #j = j+1
#if j == len(data)-1:
    #k = 0
    #dataChunk = []
    #while k < 5:
        #dataChunk.append([data[j][k][0],data[j][k][1]])
        #if k == 4:
            #dataNew.append(dataChunk)
        #k = k+1
#dataNew=np.array(dataNew)
#print(dataNew)
outfile = 'Swarm_Data'
np.savez(outfile,agent_positions=data,num_agents = 5,enclosure_size = enclosure,steps = len(data),overall_time = 10)
cap.release()
cv2.destroyAllWindows()