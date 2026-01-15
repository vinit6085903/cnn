history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)
history.history.keys()
print("Train Accuracy:", history.history['accuracy'][-1])
print("Validation Accuracy:", history.history['val_accuracy'][-1])
test_loss, test_accuracy = model.evaluate(test_data)
print("Test Accuracy:", test_accuracy)
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
# True labels
y_true = test_data.classes

# Predict
y_pred_prob = model.predict(test_data)
y_pred = np.argmax(y_pred_prob, axis=1)
print(classification_report(
    y_true,
    y_pred,
    target_names=list(test_data.class_indices.keys())
))
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=test_data.class_indices.keys(),
            yticklabels=test_data.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
plt.figure(figsize=(12,4))

# Accuracy graph
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Curve")

# Loss graph
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Curve")

plt.show()
