import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

# Load pretrained VGG16 model
model = VGG16(weights="imagenet")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to VGG16 input size
    img = cv2.resize(frame, (224, 224))

    # Convert to array & preprocess
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Predict
    preds = model.predict(img, verbose=0)

    # Decode prediction (top-1)
    label = decode_predictions(preds, top=1)[0][0]
    class_name = label[1]
    confidence = label[2] * 100

    text = f"{class_name} : {confidence:.2f}%"

    # Display prediction on original frame
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("VGG16 Real-Time Webcam", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
