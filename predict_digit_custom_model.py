import cv2
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("mnist_lenet5_finetuned.h5")

# Load and preprocess the image
img = cv2.imread("test0.png", cv2.IMREAD_GRAYSCALE)  # Read in grayscale
img = cv2.resize(img, (28, 28))  # Resize to 28x28
img = cv2.bitwise_not(img)  # Invert colors if needed (MNIST is white background)
img = img / 255.0  # Normalize
img = img.reshape(1, 28, 28, 1)  # Reshape for model input

# Make prediction
prediction = model.predict(img)
predicted_digit = np.argmax(prediction)

print(f"🔢 Predicted digit: {predicted_digit}")

img = cv2.imread("test1.png", cv2.IMREAD_GRAYSCALE)  # Read in grayscale
img = cv2.resize(img, (28, 28))  # Resize to 28x28
img = cv2.bitwise_not(img)  # Invert colors if needed (MNIST is white background)
img = img / 255.0  # Normalize
img = img.reshape(1, 28, 28, 1)  # Reshape for model input+

# Make prediction
prediction = model.predict(img)
predicted_digit = np.argmax(prediction)

print(f"🔢 Predicted digit: {predicted_digit}")
