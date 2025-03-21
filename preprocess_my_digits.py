import cv2
import numpy as np
import os

image_folder = "my_digits/"
processed_images = []
processed_labels = []

for digit in range(10):
    image_path = os.path.join(image_folder, f"my_digit_{digit}.png")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = cv2.bitwise_not(image)
    image = image / 255.0 
    processed_images.append(image.reshape(28, 28, 1))
    processed_labels.append(digit)

processed_images = np.array(processed_images)
processed_labels = np.array(processed_labels)

np.save("my_digits.npy", processed_images)
np.save("my_labels.npy", processed_labels)

print("✅ Handwritten digits preprocessed and saved!")
