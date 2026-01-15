import cv2
import numpy as np

cap = cv2.VideoCapture(0)

print("Press 'c' to capture image")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw box
    cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 2)
    cv2.imshow("Digit Recognition - LeNet5", frame)

    key = cv2.waitKey(1)

    if key == ord('c'):
        roi = frame[100:300, 100:300]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32,32))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1,32,32,1)

        prediction = model.predict(reshaped)
        digit = np.argmax(prediction)

        print("Predicted Digit:", digit)

        cv2.putText(frame, f"Digit: {digit}", (100,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Prediction", frame)
        cv2.waitKey(1000)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
