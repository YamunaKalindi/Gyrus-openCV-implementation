import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image and resize
img = cv2.imread('assets/cloud.jpg')
img = cv2.resize(img, (800, 800))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # such methods/algo work well with grayscale

# Xhi Tomasi method of extracting corners 
corners = cv2.goodFeaturesToTrack(gray,     100,            0.1, 			10)
								  # img  # num of corners  # confidence    # Eucli dist btw 2 corners

# print(corners)---> returns pixel values where corners are located (dtype = float)

corners = corners.astype(int)
 # converts the dtype to int. This is new method according to newer versios of NumPy

# Make a copy of original image for Shi-Tomasi
shi_img = img.copy()

for corner in corners:
	x, y = corner.ravel()
	cv2.circle(shi_img, (x, y), 5, (255, 0, 0), -1)

# for i in range(len(corners)):    # Loop through each corner from index 0 to last.
	#for j in range(i + 1, len(corners)):    # For each i, loop through the remaining corners (to avoid duplicate lines).
	#	corner1 = tuple(corners[i][0])
	#	corner2 = tuple(corners[j][0])
	#	color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3)))    # Generates 3 random numbers in [0, 255) → RGB
	#	cv2.line(shi_img, corner1, corner2, color, 1)

# Harris Corner Detection
harris_img = img.copy()    # copy of original image
gray_float = np.float32(gray)    # Harris requires float32 format input

harris = cv2.cornerHarris(gray_float, 2, 3, 0.04)
# img (gray) → should be float32
# blockSize = 2 (neighborhood size)
# ksize = 3 (aperture parameter of Sobel)
# k = Harris detector free parameter in the equation

harris = cv2.dilate(harris, None)    # dilate to mark the corners more clearly

# Threshold to select strong corners
mask = harris > 0.1 * harris.max()   # Boolean mask

# Further dilate the corner points to make them appear larger
mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8))  # dilate with a 5x5 kernel

# Color the dilated mask regions blue
harris_img[mask == 1] = [255, 0, 0]   # blue corners


# Convert BGR to RGB for matplotlib display
shi_img_rgb = cv2.cvtColor(shi_img, cv2.COLOR_BGR2RGB)
harris_img_rgb = cv2.cvtColor(harris_img, cv2.COLOR_BGR2RGB)

# Display both using matplotlib for side-by-side comparison
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(shi_img_rgb)
plt.title('Shi-Tomasi Corner Detection')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(harris_img_rgb)
plt.title('Harris Corner Detection')
plt.axis('off')

plt.tight_layout()
plt.show()
