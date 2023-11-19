import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

# a) LOADING AND PRE-PROCESSING THE IMAJE DATA
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_text,y_text) = mnist.load_data() # splitting into training and testing data
input_shape = (28, 28, 1)

# making sure that the values are float so that we can get decimal points after division
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value
x_train = x_train / 255
x_test = x_test / 255

print(f"shape of training : {x_train.shape}")
print(f"shape of testing : {x_test.shape}")

# b) DEFINING THE MODEL`S ARCHITECTURE
model = Sequential()
model.add(Conv2D(28, keras_size = (3,3), input_shape = input_shape))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(flatten())
model.add()
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation = 'softmax'))
model.summary()

# c) TRAINING THE MODEL
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 2)

# d) ESTIMATING THE MODEL`S PERFORMANCE
test_loss, test_acc = model.evaluate(x_test, y_text)
print("Loss=%.3f" %test_loss)
print("Accuracy=%.3f" %test_acc)

# showing image at position[] from dataset:
image = x_train[0]
plt.imshaow(np.squeeze(image), cmap = 'gray')
plt.show()

# Predicting the class of image:
image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
predict_model = model.predict([image])
print(f"Predicted class : {np.argmax(predict_model}")