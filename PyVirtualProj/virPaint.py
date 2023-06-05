import cv2
import os
import time
import HTmod as htm
import numpy as np

brushThickness = 7
eraserThickness = 25
folderPath = "VPpics"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))
header = overlayList[0]
DrawColor = (0, 0, 255)

cap = cv2.VideoCapture(0)
'''width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)'''

wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(detectionCon=0.86, maxHands=1)
xp, yp = 0, 0
imgCanvas = np.zeros((620, 880, 3), np.uint8)

while True:
    success, img = cap.read()

    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList)
        x1, y1 = lmList[8][1], lmList[8][2]  # tip of index finger
        x2, y2 = lmList[12][1], lmList[12][2]

        fingers = detector.fingersUp()
        # print(fingers)
        if fingers[1] and fingers[2]:

            ## print("Selection mode")
            if y1 < 62:
                if 100 < x1 < 180:
                    header = overlayList[0]
                    DrawColor = (0, 0, 255)
                elif 300 < x1 < 380:
                    header = overlayList[1]
                    DrawColor = (0, 255, 255)
                elif 410 < x1 < 490:
                    header = overlayList[2]
                    DrawColor = (255, 0, 0)
                elif 500 < x1 < 600:
                    header = overlayList[3]
                    DrawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), DrawColor, cv2.FILLED)

        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 7, DrawColor, cv2.FILLED)
            # print("Drawing mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if DrawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), DrawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), DrawColor, eraserThickness)

            else:
                cv2.line(img, (xp, yp), (x1, y1), DrawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), DrawColor, brushThickness)

            xp, yp = x1, y1


    img[0:62, 0:640] = header
    cv2.imshow("Image", img)
    cv2.imshow("Cnvas", imgCanvas)
    cv2.waitKey(1)
