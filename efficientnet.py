import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
output = Dense(1000, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0, preprocess_input, decode_predictions
)

# Load pretrained EfficientNetB0
model = EfficientNetB0(weights="imagenet")

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    img = cv2.resize(frame, (224, 224))

    # Preprocess for EfficientNet
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Prediction
    preds = model.predict(img, verbose=0)

    # Decode top-1 prediction
    label = decode_predictions(preds, top=1)[0][0]
    class_name = label[1]
    confidence = label[2] * 100

    text = f"{class_name} : {confidence:.2f}%"

    # Display result
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("EfficientNetB0 Real-Time Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
