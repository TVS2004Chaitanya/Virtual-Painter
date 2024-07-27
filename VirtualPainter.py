import cv2 as cv
import numpy as np
import time
import os
import HandTrackingModule as htm

brushThickness = 7
eraserThickness = 50

folderPath = "header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header = overlayList[5]

vid = cv.VideoCapture(-1)
drawColor = (33,255,33)

detector = htm.handDetector()
xp,yp=0,0
imgCanvas = np.zeros((480,640,3),np.uint8)

while True:
    # 1.Import Image
    isTrue, frame = vid.read()

    if isTrue:
        frame = cv.flip(frame, 1)
        # print(frame.shape)

        # 2.Find Hand Landmarks
        frame = detector.findHands(frame)
        lmlist = detector.findPositions(frame, draw=False)

        if len(lmlist) != 0:
            # print(lmlist)

            x1,y1 = lmlist[8][1:]  #tip of index finger
            x2,y2 = lmlist[12][1:] #tip of middle finger

            # print(x1)
            # print(f'{frame.shape}, {imgCanvas.shape}')

            fingers = detector.fingersUp()
            # print(fingers)

            if len(fingers) > 1:
                # 4.Selection Mode
                if fingers[1] and fingers[2]:
                    cv.rectangle(frame, (x1, y1-15), (x2,y2+15), drawColor, cv.FILLED)
                    # print("Selection Mode")

                    if y1 < 84:
                        if x1>130 and x1<190:
                            header = overlayList[2]
                            drawColor = (203,192,255)
                        elif x1>230 and x1<300:
                            header = overlayList[1]
                            drawColor = (0,0,255)
                        elif x1>325 and x1<390:
                            header = overlayList[3]
                            drawColor = (255,0,0)
                        elif x1>425 and x1<480:
                            header = overlayList[0]
                            drawColor = (0,255,0)
                        elif x1>520 and x1<600:
                            header = overlayList[5]
                            drawColor = (0,0,0)

        
                # 5.DrawingMode
                if fingers[1] and fingers[2] == 0:
                    cv.circle(frame, (x1,y1), 15, drawColor, cv.FILLED)
                    # print("Drawing Mode")

                    if abs(xp-x1)>30 or abs(yp-y1)>30:
                        xp,yp = x1,y1
                    
                    if drawColor != (33,255,33):
                        if drawColor==(0,0,0):
                            cv.line(frame, (xp,yp), (x1,y1), drawColor, eraserThickness)
                            cv.line(imgCanvas, (xp,yp), (x1,y1), drawColor, eraserThickness)
                    

                        cv.line(frame, (xp,yp), (x1,y1), drawColor, brushThickness)
                        cv.line(imgCanvas, (xp,yp), (x1,y1), drawColor, brushThickness)

                        xp,yp = x1,y1


        imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
        _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
        imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
        frame = cv.bitwise_and(frame, imgInv)
        frame = cv.bitwise_or(frame, imgCanvas)        

        frame[0:84, 0:640] = header
        cv.waitKey(1)
        cv.imshow("Painter", frame)
        cv.imshow("Canvas", imgCanvas)