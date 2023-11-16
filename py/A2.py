#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.datasets import mnist


# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[3]:


x_train = x_train.reshape(x_train.shape[0],28*28)
x_test = x_test.reshape(x_test.shape[0],28*28)
x_train = x_train.astype("float32")/255.0
x_test = x_test.astype('float32')/255.0


# In[4]:


from sklearn.preprocessing import LabelBinarizer


# In[5]:


lb = LabelBinarizer()#categorical to binary


# In[6]:


y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)


# In[7]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[8]:


model = Sequential()
model.add(Dense(128,input_shape=(784,),activation="sigmoid")) #imp
model.add(Dense(64,activation="sigmoid"))
model.add(Dense(10,activation="softmax"))
model.summary()


# In[9]:


from tensorflow.keras.optimizers import SGD


# In[10]:


sgd = SGD(0.01)


# In[11]:


model.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=["accuracy"])
H = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=128)


# In[12]:


predictions = model.predict(x_test,batch_size=128)


# In[13]:


from sklearn.metrics import classification_report


# In[14]:


print(classification_report(y_test.argmax(axis=1),predictions.argmax(axis=1)))


# In[15]:


import matplotlib.pyplot as plt
import numpy as np


# In[16]:


plt.plot(np.arange(0,10),H.history["loss"],label='train_loss')
plt.plot(np.arange(0,10),H.history["val_loss"],label='val_loss')
plt.plot(np.arange(0,10),H.history["accuracy"],label='accuracy')
plt.plot(np.arange(0,10),H.history["val_accuracy"],label='val_accuracy')
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:




