import cv2
import numpy as np
import time
import HTmod as htm
import autopy

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
frameR = 100
pTime = 0
smoothening = 8
plocX, plocY = 0, 0
clocX, clocY = 0, 0
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)
detector = htm.handDetector(maxHands=1)

while True:
     success, img = cap.read()
     img = cv2.flip(img, 1)
     img = detector.findHands(img)
     lmList, bbox = detector.findPosition(img)

     if len(lmList) != 0:
         x1, y1 = lmList[8][1:]
         x2, y2 = lmList[12][1:]
         # print(x1, y1, x2, y2)

         fingers = detector.fingersUp()
         # print(fingers)
         cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (204, 0, 0), 2)

         if fingers[1] == 1 and fingers[2] == 0:

             x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
             y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

             clocX = plocX + (x3 - plocX) / smoothening
             clocY = plocY + (y3 - plocY) / smoothening


             autopy.mouse.move(x3, y3)
             cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
             plocX, plocY = clocX, clocY

         if fingers[1] == 1 and fingers[2] == 1:
             length, img, lineInfo = detector.findDistance(8, 12, img)
             # print(length)
             if length < 40:
                 cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (51, 0, 102), cv2.FILLED)
                 autopy.mouse.click()







     cTime = time.time()
     fps = 1 / (cTime - pTime)
     pTime = cTime
     cv2.putText(img, f'FPS:{(int(fps))}', (5, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (204, 0, 0), 2)
     cv2.imshow("Image", img)
     cv2.waitKey(1)