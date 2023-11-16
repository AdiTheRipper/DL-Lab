#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


# In[2]:


#grabbing CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0
# train_images


# In[3]:


#showing images of mentioned categories
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
#     plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()


# In[4]:


#building CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()


# In[5]:


#model compilation
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
epochs = 10
h = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))


# In[ ]:





# In[6]:


import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras import datasets  


model = keras.Sequential([
    keras.layers.Flatten(input_shape= (32,32,3)),
    keras.layers.Dense(128,activation = "relu"),
    keras.layers.Dense(64,activation = "relu"),
    keras.layers.Dense(32,activation = "relu"),
    keras.layers.Dense(10,activation="softmax")
])
model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

history = model.fit(train_images,train_labels,validation_data=(test_images,test_labels),epochs = 11)

test_loss,test_acc = model.evaluate(test_images,test_labels)
print("loss %.3f"%test_loss)
print("acc %.3f"%test_acc)

predicted_values = model.predict(test_images)
predicted_values.shape


# In[7]:


n = random.randint(0,9999)
plt.figure(figsize=(10,10))
plt.imshow(test_images[n])
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.title(class_names[np.argmax(predicted_values[n])])


# In[ ]:





# In[ ]:




