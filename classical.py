import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
--0--
|   |
3   4
|   |
--1--
|   |
5   6
|   |
--2--
"""

decipher = {}
decipher[125] = '0' # 023456
decipher[80] = '1' # 46
decipher[55] = '2' # 01245
decipher[87] = '3' # 01246
decipher[90] = '4' # 1346
decipher[79] = '5' # 01236
decipher[111] = '6' # 012356
decipher[81] = '7' # 046
decipher[127] = '8' # 0123456
decipher[95] = '9' # 012346

e = 0.3
segments = [
	[(0 + e,  0), (1 - e, 0.2)],
	[(0 + e,0.4), (1 - e, 0.6)],
	[(0 + e,0.8), (1 - e, 1.0)],
	[(0,0), (0.3, 0.5)],
	[(0.7,0), (1.0, 0.5)],
	[(0,0.5), (0.3, 1.0)],
	[(0.7,0.5), (1.0, 1.0)]
]
threshold = 500
color = (255)

def get_points(idx, shape):
	(h,w) = shape
	pt1 = (int(segments[idx][0][0] * w), int(segments[idx][0][1] * h))
	pt2 = (int(segments[idx][1][0] * w), int(segments[idx][1][1] * h))
	return pt1, pt2

def get_key(arr):
	key = 0
	# print(arr)
	for i in range(len(arr)):
		# print("K", 2**(i) * (arr[i] > threshold))
		# print(arr[i] > threshold),
		key += 2**(i) * (arr[i] > threshold)

	# print("KEY", key)
	return key

def decode_segments(image):
	_ ,bin = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
	(h,w) = bin.shape

	hits = []
	for i in range(len(segments)):
		pt1, pt2 = get_points(i, bin.shape)
		cv2.rectangle(image, pt1, pt2, color, 5)
		roi = bin[pt1[1]: pt2[1],pt1[0]: pt2[0]]
		
		n_white_pix = np.sum(roi == 255)
		# print(n_white_pix)
		hits.append(n_white_pix)

	key = get_key(hits)
	# print(key)
	plt.imshow(image)
	plt.title(decipher[key])
	plt.show()


image = cv2.imread('images/text.jpg', 0)
y_t = 930
y_b = 1100
kernel = np.ones((3,3),np.uint8)
digit = 1
for x in range(100, 1420, 153):

	# Crop and resize number
	roi = image[y_t:y_b,x:x+120]
	decode_segments(roi)