import cv2

image = cv2.imread('img.jpg')
image[:,:,0] = 0
image[:,:,1] = 0
image[:,:,2] = 0
cv2.imshow('image', image)
cv2.waitKey(0)