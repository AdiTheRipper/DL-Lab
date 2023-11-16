# Import necessary packages
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_test = x_test.reshape(x_test.shape[0], 28 * 28)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype('float32') / 255.0

# Binarize the labels
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# Define the network architecture
model = Sequential()
model.add(Dense(128, input_shape=(784,), activation="sigmoid"))
model.add(Dense(64, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

# Display the model summary
model.summary()

# Compile the model with Stochastic Gradient Descent (SGD)
sgd = SGD(0.01)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
H = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128)

# Make predictions on the test set
predictions = model.predict(x_test, batch_size=128)

# Evaluate the network
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))

# Plot the training loss and accuracy
plt.plot(np.arange(0, 10), H.history["loss"], label='train_loss')
plt.plot(np.arange(0, 10), H.history["val_loss"], label='val_loss')
plt.plot(np.arange(0, 10), H.history["accuracy"], label='accuracy')
plt.plot(np.arange(0, 10), H.history["val_accuracy"], label='val_accuracy')
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
