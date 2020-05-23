import tensorflow.keras as keras 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load model
model = keras.models.load_model('model/mnist.hdf5')

def pred(roi28, sol):
	
	# Transform image into tensor
	image = roi28[np.newaxis,:,:,np.newaxis]

	# Predict digit
	prediction = model.predict(image)
	# prediction = model.predict_classes(image)
	print(prediction)
	best_guess = np.argmax(prediction[0])

	# Plot result
	plt.imshow(roi28)
	title = "Guess: " + str(best_guess) + " | Solution: " + str(sol)
	plt.title(title)
	plt.show()
	print(title)


# Test mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# for i in range(10):
# 	pred(x_test[i,:], y_test[i])

# Test own dataset
image = cv2.imread('images/text.jpg', 0)
y_t = 920
y_b = 1120
kernel = np.ones((3,3),np.uint8)
digit = 1
for x in range(120, 1420, 150):

	# Crop and resize number
	roi = image[y_t:y_b,x:x+130]
	roi28 = cv2.resize(roi, (28, 28))

	# Binarize image
	# ret1,roi28 = cv2.threshold(roi28,127,255,cv2.THRESH_BINARY)

	# Dilate image
	
	roi28 = cv2.dilate(roi28,kernel,iterations = 1)
	pred(roi28, digit)
	digit += 1
	


