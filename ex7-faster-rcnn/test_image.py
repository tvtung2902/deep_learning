import cv2

#blue green red
image = cv2.imread('../all-data/volleyball/maJM7QoN7-2935.532_2.jpg')
height, width, _ = image.shape

with open('../all-data/volleyball/maJM7QoN7-2935.532_2.txt', 'r') as fo:
    annotations = fo.readlines()

annotation = [anno.split() for anno in annotations]

for anno in annotation:
    category, x, y, w, h = anno
    x_center = float(x) * width
    y_center = float(y) * height
    w = float(w) * width
    h = float(h) * height
    x_tl = int(x_center - w / 2)
    y_tl = int(y_center - h / 2)
    x_br = int(x_center + w / 2)
    y_br = int(y_center + h / 2)
    color = (0, 0, 255) if category.__eq__('1') else (255, 0, 0)
    cv2.rectangle(image, (x_tl, y_tl), (x_br, y_br), color, 2)

cv2.imshow('image', image)
cv2.waitKey(0)