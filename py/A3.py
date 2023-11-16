#!/usr/bin/env python
# coding: utf-8

# Importing the necessary libraries
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import random
import numpy as np

# Loading and preprocessing the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# # Displaying images of specific categories
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
# plt.figure(figsize=(10, 10))
# for i in range(10):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(train_images[i])
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

# # Building the CNN model
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10))

# # Displaying the model summary
# model.summary()

# # Compiling the model
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# # Training the model
# epochs = 10
# history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))

# Another model definition and training
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=11)

# Evaluating the model's performance on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Loss: %.3f" % test_loss)
print("Test Accuracy: %.3f" % test_acc)

# Predicting and displaying a random test image with its predicted class
n = random.randint(0, 9999)
plt.figure(figsize=(10, 10))
plt.imshow(test_images[n])
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.title("Predicted Class: " + class_names[np.argmax(model.predict(np.expand_dims(test_images[n], axis=0)))])
plt.show()
