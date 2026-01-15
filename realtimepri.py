import cv2
import numpy as np
import tensorflow as tf

# Load trained AlexNet model
model = tf.keras.models.load_model("alexnet_model.h5")

# Class names (example: CIFAR-10)
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for AlexNet
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    predictions = model.predict(img, verbose=0)
    class_id = np.argmax(predictions)
    confidence = np.max(predictions)

    label = f"{class_names[class_id]} ({confidence*100:.2f}%)"

    # Show prediction on frame
    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Real-Time AlexNet Prediction", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
