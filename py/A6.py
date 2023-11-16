#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.__version__


# In[2]:


img_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    # rotation_range=90,
    brightness_range=(0.5, 1),
    # shear_range=0.2,
    # zoom_range=0.2,
    channel_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255,
    validation_split=0.3
)


# In[3]:


root_dir = './Downloads/archiveMybro/caltech-101'

img_generator_flow_train = img_generator.flow_from_directory(
    directory=root_dir,
    target_size=(224, 224),
    batch_size=32,
    shuffle=True,
    subset="training"
)

img_generator_flow_valid = img_generator.flow_from_directory(
    directory=root_dir,
    target_size=(224, 224),
    batch_size=32,
    shuffle=True,
    subset="validation"
)


# In[4]:


imgs, labels = next(iter(img_generator_flow_train))
for img, label in zip(imgs, labels):
  plt.imshow(img)
  plt.show()


# In[5]:


base_model = tf.keras.applications.InceptionV3(input_shape=(224,224,3),
include_top=False,
weights = "imagenet"
)


# In[6]:


base_model.trainable = False


# In[7]:


model = tf.keras.Sequential([
base_model,
tf.keras.layers.MaxPooling2D(),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(102, activation="softmax")
])


# In[8]:


model.summary()


# In[9]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
loss = tf.keras.losses.CategoricalCrossentropy(),
metrics = [tf.keras.metrics.CategoricalAccuracy()])


# In[10]:


model.fit(img_generator_flow_train, 
          validation_data=img_generator_flow_valid, 
          steps_per_epoch=20, 
          epochs=5)


# In[11]:


plt.plot(model.history.history["categorical_accuracy"], c="r", label="train_accuracy")
plt.plot(model.history.history["val_categorical_accuracy"], c="b", label="test_accuracy")
plt.legend(loc="upper left")
plt.show()


# In[12]:


base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
loss = tf.keras.losses.CategoricalCrossentropy(),
metrics = [tf.keras.metrics.CategoricalAccuracy()])
model.fit(img_generator_flow_train, validation_data=img_generator_flow_valid, steps_per_epoch=20, epochs=5)

