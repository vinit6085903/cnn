import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def predict_uploaded_image(img_path):
    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Image not found!")
        return

    # Resize to 32x32
    img = cv2.resize(img, (32, 32))

    # Normalize
    img = img / 255.0

    # Reshape for model
    img = img.reshape(1, 32, 32, 1)

    # Predict
    prediction = model.predict(img)
    digit = np.argmax(prediction)

    # Show image
    plt.imshow(img.reshape(32,32), cmap='gray')
    plt.title(f"Predicted Digit: {digit}")
    plt.axis("off")
    plt.show()

# Example
predict_uploaded_image("digit.png")
