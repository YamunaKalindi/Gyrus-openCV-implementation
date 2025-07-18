import numpy as np
import cv2

cap = cv2.VideoCapture('assets/video1.mp4')

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (400, 400))
    width = int(cap.get(3))
    height = int(cap.get(4))

    img = cv2.line(frame, (0, 0), (width, height), (255, 0, 0), 10)
                        # start     # end           # color   # thickness
    img = cv2.line(img, (0, height), (width, 0), (0, 255, 0), 5)
    img = cv2.rectangle(img, (100, 100), (200, 200), (128, 128, 128), 5)
    img = cv2.circle(img, (300, 300), 60, (0, 0, 255), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX # font style
    img = cv2.putText(img, 'Tim is Great!', (10, height - 30), font, 2, (255, 255, 255), 5, cv2.LINE_AA)

    cv2.imshow('frame', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()