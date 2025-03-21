import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("mnist_lenet5_finetuned.h5")

img = cv2.imread("test0.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = cv2.bitwise_not(img)
img = img / 255.0
img = img.reshape(1, 28, 28, 1)

prediction = model.predict(img)
predicted_digit = np.argmax(prediction)

print(f"🔢 Predicted digit: {predicted_digit}")

img = cv2.imread("test1.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = cv2.bitwise_not(img)
img = img / 255.0 
img = img.reshape(1, 28, 28, 1) 

prediction = model.predict(img)
predicted_digit = np.argmax(prediction)

print(f"🔢 Predicted digit: {predicted_digit}")
