import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("mnist_lenet5.h5")

x_train = np.load("my_digits.npy")
y_train = np.load("my_labels.npy")

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5, batch_size=2)

model.save("mnist_lenet5_finetuned.h5")

print("✅ Model fine-tuned and saved as 'mnist_lenet5_finetuned.h5'")