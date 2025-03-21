import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("mnist_lenet5.h5")

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

predictions = model.predict(x_test)

index = 0
plt.imshow(x_test[index].reshape(28,28), cmap='gray')
plt.title(f"Predicted: {np.argmax(predictions[index])}, Actual: {y_test[index]}")
plt.show()
