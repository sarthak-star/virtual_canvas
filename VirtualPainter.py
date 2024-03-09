import cv2 
import time
import numpy as np
import os
import HandTrackingModule as htm

brushThickness = 25
eraserThickness = 100

folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0, 255)


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector()
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    success , img = cap.read()
    img = cv2.flip(img,1)
    
    img = detector.findHands(img)
    lmlist = detector.findPositions(img,draw=False)
    
    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]
        
        fingers = detector.fingersUp()
        
        if fingers[1] and fingers[2]:
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
        
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            
            
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            
            xp, yp = x1, y1

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img,imgInv)
        img = cv2.bitwise_or(img,imgCanvas)
        
        # img[0:78, 0:650] = header
        cv2.imshow("Image", img)
        # cv2.imshow("Canvas", imgCanvas)
        # cv2.imshow("Inv", imgInv)
        cv2.waitKey(1)