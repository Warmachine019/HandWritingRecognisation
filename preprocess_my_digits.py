import cv2
import numpy as np
import os

# Folder where you stored your handwritten digit images
image_folder = "my_digits/"
processed_images = []
processed_labels = []

# Loop through each digit image (0-9)
for digit in range(10):
    image_path = os.path.join(image_folder, f"my_digit_{digit}.png")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    image = cv2.resize(image, (28, 28))  # Resize to MNIST size
    image = cv2.bitwise_not(image)  # Invert colors if needed
    image = image / 255.0  # Normalize
    processed_images.append(image.reshape(28, 28, 1))  # Reshape for model
    processed_labels.append(digit)  # Store the correct label

# Convert to NumPy arrays
processed_images = np.array(processed_images)
processed_labels = np.array(processed_labels)

# Save the data for training
np.save("my_digits.npy", processed_images)
np.save("my_labels.npy", processed_labels)

print("✅ Handwritten digits preprocessed and saved!")
