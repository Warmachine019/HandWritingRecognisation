# Handwritten Digit Recognition

This project is a **Handwritten Digit Recognition** system based on a **Convolutional Neural Network (CNN)**, trained on the **MNIST dataset** and fine-tuned using custom handwritten digits.

## 📌 Features
- Train a CNN model using the **MNIST dataset**.
- Fine-tune the model with **custom handwritten digits**.
- Predict handwritten digits from image files.
- Uses **TensorFlow/Keras** for deep learning and **OpenCV** for image preprocessing.

## 📁 Project Structure
```
📂 HandWritingRecognisation
│── fine_tune_model.py       # Fine-tune the model with custom images
│── load_model.py            # Load the trained model
│── mnist_cnn.py             # CNN model for digit recognition
│── mnist_test.py            # Test the model with sample MNIST data
│── my_digits.npy            # Custom handwritten digits dataset
│── my_labels.npy            # Labels for custom handwritten digits
│── predict_digit.py         # Script to predict digits from images
│── preprocess_my_digits.py  # Preprocessing script for custom images
│── test.png                 # Sample handwritten digit image
│── /my_digits/              # Handwritten images to train the model on
```

## 🚀 How to Run

### 1️⃣ Install Dependencies
Make sure you have Python installed and then install the required libraries:
```bash
pip install tensorflow opencv-python numpy matplotlib
```

### 2️⃣ Train the Model
Run the following script to train the CNN model on the **MNIST dataset**:
```bash
python mnist_cnn.py
```
This will create a trained model (`mnist_lenet5.h5`).

### 3️⃣ Fine-Tune with Custom Handwriting
If you want to **improve accuracy** on your own handwriting, fine-tune the model:
```bash
python fine_tune_model.py
```
This will generate a **fine-tuned model** (`mnist_lenet5_finetuned.h5`).

### 4️⃣ Predict the digits
To predict a digit from an image (e.g., `test.png`):
```bash
python predict_digit.py
```

## 🔍 Troubleshooting
- If the model predicts the wrong digit consistently, try fine-tuning it with more **handwritten samples**.
- Ensure the image file exists and is correctly **preprocessed**.
- If OpenCV throws a file error, verify the **image path**

## 💻 Update:
1. "predict_digit_custom_model.py" now scans 2 images at once, "test0.png" and "test1.png" and returns the predicted value for both images.
2. Add more reference images to the "my_digits" folder and train the model in order to get more accurate results.
