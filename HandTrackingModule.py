import cv2
import mediapipe as mp
import time
import math
import numpy as np

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectCon=0.5,trackCon=0.5,modelC=1):
        self.modelC=modelC
        self.mode=mode
        self.maxHands=maxHands
        self.detectCon=detectCon
        self.trackCon=trackCon
        self.mphands = mp.solutions.hands
        self.mpdraw =  mp.solutions.drawing_utils
        self.hands = self.mphands.Hands(
            self.mode,
            self.maxHands,
            self.modelC,
            self.detectCon,
            self.trackCon
        )
        self.tipIDS = [4,8,12,16,20]
    
    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img,handLMS,self.mphands.HAND_CONNECTIONS)
        return img

    def findPositions(self,img,handNo=0,draw=True):
        self.lmlist =[]
        
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myhand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                
                self.lmlist.append([id,cx,cy])
                
                if draw:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        return self.lmlist
    def fingersUp(self):
        fingers = []
        
        if self.lmlist[self.tipIDS[0]][1] > self.lmlist[self.tipIDS[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if self.lmlist[self.tipIDS[id]][2] < self.lmlist[self.tipIDS[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        
        return fingers
        
        
def main():
    ptime=0
    ctime=0
    
    cap=cv2.VideoCapture(0)
    detector=handDetector()
    
    while True:
        success,img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPositions(img)
        
        if len(lmlist) != 0:
            print(lmlist[4])
        
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        
        cv2.putText(img , str(int(fps)) , (10,70) , cv2.FONT_HERSHEY_PLAIN , 3 , (255,0,255) , 3)
        cv2.imshow("Image",img)
        cv2.waitKey(1)
    
    
    
if __name__ == "__main__":
    main()        