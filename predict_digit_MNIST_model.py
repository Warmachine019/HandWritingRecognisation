import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model("mnist_lenet5.h5")

# Load the MNIST dataset (only test set for predictions)
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data (normalize & reshape)
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# Make a prediction
predictions = model.predict(x_test)

# Pick a test image
index = 0  # Change this number to see different test images
plt.imshow(x_test[index].reshape(28,28), cmap='gray')
plt.title(f"Predicted: {np.argmax(predictions[index])}, Actual: {y_test[index]}")
plt.show()
