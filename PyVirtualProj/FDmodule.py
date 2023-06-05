import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon

        self.mpFaceDect = mp.solutions.face_detection
        self.faceDect = self.mpFaceDect.FaceDetection(self.minDetectionCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDect.process(imgRGB)
        # print (self.results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([bbox, id, detection.score])
                cv2.rectangle(img, bbox, (28, 249, 24), 1)
                # cv2.putText(img, f'{(int(detection.score[0] * 100))}%', (bbox[0], bbox[1] - 20), (5, 30), cv2.FONT_HERSHEY_DUPLEX,1, (9, 9, 255), 1)
        return img, bboxs


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        # print(bboxs)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {(int(fps))}', (5, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (9, 9, 255), 1)
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
