import cv2
import random

img = cv2.imread('assets/cloud.jpg', -1) # -1 represents including transperancy channel

print(img.shape)    #will give num of rows, col and NUM OF CHANNELS, that means-----> img.shape = (height, width, channels)


# Change first 100 rows to random pixels
for i in range(100):
	for j in range(img.shape[1]):  #img.shape = (height, width, channels), to modify width/col we say shape[1]
		img[i][j] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
						#blue					#green					#red

# Copy part of image
tag = img[1800:2000, 1600:2000]   # the dimension of tag and the portion in the image should be same
img[800:1000, 600:1000] = tag

img = cv2.resize(img, (400, 400))
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()