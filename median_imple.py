import cv2
import numpy as np

# Read a noisy image
# salt and pepper noise--> a type of noise that appears in digital images as randomly scattered 
# black and white pixels, resembling salt and pepper sprinkled on the image
img = cv2.imread(r'C:\Users\achyu\Downloads\OpenCV-Tutorials-main\OpenCV-Tutorials-main\assets\noisy_image.png')

if img is None:
    print("Image not loaded. Check the path again.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply median filter --> to smoothen the image and reduse noise
median_filtered = cv2.medianBlur(gray, 5)

# Pick a region in the center
start_row = gray.shape[0] // 2 - 2
start_col = gray.shape[1] // 2 - 2

print("Original pixel values (5x5 patch):")
print(gray[start_row:start_row+5, start_col:start_col+5])

print("After median filtering (5x5 patch):")
print(median_filtered[start_row:start_row+5, start_col:start_col+5])

# Show before and after
cv2.imshow("Original Grayscale", gray)
cv2.imshow("Median Filtered", median_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
