import cv2

img = cv2.imread('assets/cloud.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints
keypoints = sift.detect(gray, None)

# Draw keypoints
img_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


img_keypoints = cv2.resize(img_keypoints, (900, 900))
cv2.imshow("SIFT Keypoints", img_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
