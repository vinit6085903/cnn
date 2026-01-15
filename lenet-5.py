import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Resize from 28x28 to 32x32 (LeNet-5 requirement)
x_train = tf.image.resize(x_train[..., np.newaxis], (32, 32))
x_test = tf.image.resize(x_test[..., np.newaxis], (32, 32))

# One-hot encoding labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
model = models.Sequential([
    layers.Conv2D(6, kernel_size=(5,5), activation='tanh', input_shape=(32,32,1)),
    layers.AveragePooling2D(),

    layers.Conv2D(16, kernel_size=(5,5), activation='tanh'),
    layers.AveragePooling2D(),

    layers.Flatten(),

    layers.Dense(120, activation='tanh'),
    layers.Dense(84, activation='tanh'),
    layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1
)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)
def predict_digit(image):
    """
    image: 28x28 grayscale image
    """
    image = image / 255.0
    image = tf.image.resize(image[..., np.newaxis], (32, 32))
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    return np.argmax(prediction)
index = 0
plt.imshow(x_test[index].numpy().reshape(32,32), cmap='gray')
plt.title(f"Predicted: {np.argmax(model.predict(x_test[index:index+1]))}")
plt.show()
