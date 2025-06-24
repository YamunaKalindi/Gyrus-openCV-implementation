import numpy as np
import cv2

cap = cv2.VideoCapture('assets/video1.mp4')

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # HSV --> Hue, Saturation, Brightness. Makes it easier to detect colors
    
    # lower_blue = np.array([90, 50, 50])     
    # upper_blue = np.array([130, 255, 255])

    # for my video, orange suits
    lower_orange = np.array([10, 100, 100])     # thresholds in HSV space.
    upper_orange = np.array([25, 255, 255])


    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # This returns a black & white image. White (255) --->  HSV pixel falls in blue range; Black (0) elsewhere
    # Mask isolates the blue areas in the image.

    result = cv2.bitwise_and(frame, frame, mask=mask)

    # redize both because, they are too big to fit
    result = cv2.resize(result, (400, 400))
    mask = cv2.resize(mask, (400, 400))


    cv2.imshow('frame', result)
    cv2.imshow('mask', mask)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
