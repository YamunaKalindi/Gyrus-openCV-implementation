import cv2
import numpy as np

# Step 1: Load apple and orange images
apple = cv2.imread('assets/apple.jpg')
orange = cv2.imread('assets/orange.jpg')

# Set fixed size for both images
fixed_size = (400, 400)  # (width, height)

# Resize both images to the same size
apple = cv2.resize(apple, fixed_size)
orange = cv2.resize(orange, fixed_size)

# Crop both to same width (min of both)
min_width = min(apple.shape[1], orange.shape[1])
apple = apple[:, :min_width]
orange = orange[:, :min_width]

# Step 2: Create left-right composite (half apple, half orange)
mid = min_width // 2
composite = np.hstack((apple[:, :mid], orange[:, mid:]))

# Step 3: Simulate motion by slightly shifting the composite
frame1_gray = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
frame2 = np.roll(composite, shift=2, axis=1)  # small horizontal shift
frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Step 4: Detect keypoints in frame1
features = cv2.goodFeaturesToTrack(frame1_gray, maxCorners=200, qualityLevel=0.01, minDistance=10)

# Step 5: Apply Lucas-Kanade Optical Flow
features = np.float32(features)
features_next, status, error = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, features, None)

# Step 6: Draw arrows on composite image
result_img = composite.copy()
for i, (pt1, pt2) in enumerate(zip(features, features_next)):
    if status[i]:
        x1, y1 = pt1.ravel()
        x2, y2 = pt2.ravel()
        cv2.arrowedLine(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, tipLength=0.3)

# Resize for display
result_img = cv2.resize(result_img, (800, 400))

# Convert to grayscale before showing
gray_result = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Lucas-Kanade Optical Flow (Grayscale)", gray_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
