import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode=False, upBody=False, modelC=1, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.modelC = modelC
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.modelC, self.smooth, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, Draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if Draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmList=[]
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                 # print(id,lm)
                 h, w, c = img.shape
                 cx, cy = int(lm.x * w), int(lm.y * h)
                 lmList.append([id,cx,cy])
                 # print(id, cx, cy)
                 # if id == 0:
                 if draw:
                     cv2.circle(img, (cx, cy), 3, (229, 25, 66), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, True)
        # print(lmList) # PRINTS THE LIST OF LANDMARKS

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS:{(int(fps))}', (5, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (9, 9, 255), 1)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()