import cv2
import matplotlib.pyplot as plt

y_t = 920
y_b = 1120
image = cv2.imread('images/text.jpg', 0)

digit = 1
for x in range(120, 1420, 150):
	roi = image[y_t:y_b,x:x+130]
	roi28 = cv2.resize(roi, (28, 28))
	cv2.imshow(str(digit), roi28)
	cv2.waitKey()
	digit += 1