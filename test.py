import tensorflow.keras as keras 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

model = keras.models.load_model('model/mnist4')

image = cv2.imread('images/2.jpeg', 0)

# predictions = model.predict(image[np.newaxis,:,:,np.newaxis])
# print(predictions)

def pred(roi28, sol):
	kernel = np.ones((4,4),np.uint8)

	roi28 = cv2.dilate(roi28,kernel,iterations = 1)

	# print(roi28.shape)
	# plt.imshow(roi28)
	# plt.show()
	image = roi28[np.newaxis,:,:,np.newaxis]
	# print("minmax", np.amin(roi28), np.amax(roi28))
	# print(roi28.shape)
	
	# digit += 1
	prediction = model.predict_classes(image)
	max_index = prediction[0]
	# max_index = np.argmax(prediction[0], axis=0)
	print(max_index, sol)
	print(type(str(max_index)))
	
	
	# cv2.putText(roi28, sol, org, font,  
 #                   fontScale, color, thickness, cv2.LINE_AA) 
	plt.imshow(roi28)
	plt.show()
	print("I guess it is:  " + str(max_index))

y_t = 920
y_b = 1120
org = (20, 20) 
font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 1
   
# Blue color in BGR 
color = (255) 
  
# Line thickness of 2 px 
thickness = 2

image = cv2.imread('images/text.jpg', 0)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# for i in range(10):
# 	pred(x_test[i,:], y_test[i])

digit = 1
for x in range(120, 1420, 150):
	roi = image[y_t:y_b,x:x+130]
	print(roi.shape)

	roi28 = cv2.resize(roi, (28, 28))
	ret1,th1 = cv2.threshold(roi28,127,255,cv2.THRESH_BINARY)

	# roi28 /= 255
	pred(th1, digit)
	


