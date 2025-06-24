import numpy as np
import cv2

cap = cv2.VideoCapture('assets/video1.mp4') # in case of webcam, gets control over the web cam for execution purpose
# cv2.VideoCapture(0) indicates live video feed

# Resize target (final output will be 400x400)
target_width = 400
target_height = 400

while True:
    ret, frame = cap.read()  # reads frame by frame
   
    # frame = cv2.resize(frame, (400,400))   ----> this is before layouting

        # Resize the original frame to 400x400
    frame = cv2.resize(frame, (target_width, target_height))

     # extract the resized width and height
    height, width = frame.shape[:2]  # height = 400, width = 400

    image = np.zeros(frame.shape, np.uint8)

    smaller_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    image[:height//2, :width//2] = cv2.rotate(smaller_frame, cv2.ROTATE_180)
    image[height//2:, :width//2] = smaller_frame
    image[:height//2, width//2:] = cv2.rotate(smaller_frame, cv2.ROTATE_180)
    image[height//2:, width//2:] = smaller_frame
    
    cv2.imshow('frame', image)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()