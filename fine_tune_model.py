import tensorflow as tf
import numpy as np  # Also ensure NumPy is imported

# Load the model
model = tf.keras.models.load_model("mnist_lenet5.h5")


# Load dataset
x_train = np.load("my_digits.npy")
y_train = np.load("my_labels.npy")  # These are integers (0-9)

# 🚀 Fix: DO NOT One-Hot Encode Labels
# y_train = tf.keras.utils.to_categorical(y_train, num_classes=10) ❌ REMOVE THIS LINE

# 🚀 Fix: Compile with sparse_categorical_crossentropy
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Fine-tune the model
model.fit(x_train, y_train, epochs=5, batch_size=2)

# Save the fine-tuned model
model.save("mnist_lenet5_finetuned.h5")

print("✅ Model fine-tuned and saved as 'mnist_lenet5_finetuned.h5'")
