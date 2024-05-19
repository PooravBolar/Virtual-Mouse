import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import autopy

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

frameRed = 100

# Size of your desktop screen
wScreen, hScreen = 1920, 1080
# Size of camera
wCamera, hCamera = 640, 480

detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.6, minTrackCon=0.6)

# Lists to keep track of previous positions for smoothing
smoothening = 7
prev_locations_x, prev_locations_y = [], []

while True:
    success, img = cap.read()

    hands, img = detector.findHands(img, draw=True)

    if hands:
        hand = hands[0]
        bbox = hand['bbox']
        center = hand['center']
        handType = hand['type']
        lmList = hand['lmList']

        fingers = detector.fingersUp(hand)

        cv2.rectangle(img, (frameRed, frameRed), (wCamera - frameRed, hCamera - frameRed), (255, 0, 255), 2)

        if fingers == [0, 1, 0, 0, 0]:
            # Convert coordinates
            x1 = np.interp(lmList[8][0], (frameRed, wCamera - frameRed), (0, wScreen))
            y1 = np.interp(lmList[8][1], (frameRed, hCamera - frameRed), (0, hScreen))

            # Add the current location to the lists
            prev_locations_x.append(x1)
            prev_locations_y.append(y1)

            # Maintain only the last 'smoothening' number of positions
            if len(prev_locations_x) > smoothening:
                prev_locations_x.pop(0)
                prev_locations_y.pop(0)

            # Calculate the average of the positions
            avg_x = np.mean(prev_locations_x)
            avg_y = np.mean(prev_locations_y)

            # Move mouse
            autopy.mouse.move(wScreen - avg_x, avg_y)
            cv2.circle(img, (lmList[8][0], lmList[8][1]), 15, (255, 0, 255), cv2.FILLED)

        if fingers == [0, 1, 1, 0, 0]:
            length, info, img = detector.findDistance(lmList[8][0:2], lmList[12][0:2], img, draw=True)

            if length < 25:
                cv2.circle(img, (lmList[8][0], lmList[8][1]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    cv2.imshow("Img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
