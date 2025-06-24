import cv2

img = cv2.imread('assets/cloud.jpg', 1) # 1 represents to read img in color (default)

# reduce the size to half it's original:
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.imwrite('new_img.jpg', img)
img = cv2.resize(img, (400,400))

# img_back = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)---> If I want to see the original image

cv2.imshow('Image', img)

# cv2.imshow('Image', img_back)
cv2.waitKey(0)
cv2.destroyAllWindows()