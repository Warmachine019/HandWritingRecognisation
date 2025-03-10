import tensorflow as tf
import numpy as np
import cv2  # OpenCV for image processing

# Load the saved model
model = tf.keras.models.load_model("mnist_lenet5.h5")

# Load the image (update the filename if needed)
image_path = "test5.png"  # Update this to match your file name
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale

# Convert to binary image (thresholding) to remove noise
_, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

# Resize to 28x28 (same as MNIST dataset)
image = cv2.resize(image, (28, 28))

# Normalize pixel values (0-1 range) & reshape for model input
image = image / 255.0
image = image.reshape(1, 28, 28, 1)  # Add batch dimension

# Predict the digit
prediction = model.predict(image)
predicted_digit = np.argmax(prediction)

print(f"Predicted Digit: {predicted_digit}")
