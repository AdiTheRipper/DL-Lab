#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
import pandas as pd;
from sklearn.preprocessing import MinMaxScaler 
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout


# In[2]:


raw_data = pd.read_csv('/home/aditya/Downloads/creditcard.csv')
raw_data


# In[ ]:


raw_data = raw_data.drop("Time",axis=1)


# In[ ]:


raw_data.Class.unique()
features = raw_data.drop("Class",axis=1)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(features,raw_data['Class'],random_state=4,test_size=0.3)


# In[ ]:


train_data = x_train.loc[y_train[y_train==1].index]


# In[ ]:


minmax = MinMaxScaler(feature_range=(0, 1))
x_train_scaled = minmax.fit_transform(train_data)
x_test_scaled = minmax.transform(x_test)


# In[ ]:


class AutoEncoder(Model):
    def __init__(self,output_unit,ldim=8):
        super().__init__()
        self.encoder = Sequential([
#             Dense(64,activation='relu'),
#             Dropout(0.1),
#             Dense(32,activation='relu'),
#             Dropout(0.1),
            Dense(16,activation='relu'),
            Dropout(0.1),
            Dense(ldim,activation='relu')
        ])
        self.decoder = Sequential([
            Dense(16,activation='relu'),
#             Dropout(0.1),
#             Dense(32,activation='relu'),
#             Dropout(0.1),
#             Dense(64,activation='relu'),
            Dropout(0.1),
            Dense(output_unit,activation='sigmoid')
        ])
    def call(self,inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


# In[ ]:


model = AutoEncoder(output_unit =  x_train_scaled.shape[1])
model.compile(optimizer='adam',loss='msle',metrics=['mse'])
h = model.fit(
    x_train_scaled,
    x_train_scaled,
    validation_data=(x_test_scaled,x_test_scaled),epochs=20,batch_size=512
)
print(x_train_scaled.shape[1])


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.plot(h.history['loss'])
plt.plot(h.history['mse'])
plt.plot(h.history['val_loss'])
plt.plot(h.history['val_mse'])
plt.legend(['loss','val_loss'])
plt.xlabel('Epochs')
plt.ylabel('MSLE Loss')
plt.show()


# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd


# In[ ]:


def find_threshold(model,x_train_scaled):
    recons = model.predict(x_train_scaled)
    recons_error = tf.keras.metrics.msle(recons,x_train_scaled)
    threshold = np.mean(recons_error.numpy()) + np.std(recons_error.numpy())
    return threshold


# In[ ]:


def get_pred(model,x_test_scaled,threshold):
    pred = model.predict(x_test_scaled)
    error = tf.keras.metrics.msle(pred,x_test_scaled)
    AnomalyMask = pd.Series(error)>threshold
    return AnomalyMask.map(lambda x:0.0 if x==True else 1.0)


# In[ ]:


threshold = find_threshold(model,x_train_scaled)
print(threshold)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


pred = get_pred(model,x_test_scaled,threshold)


# In[ ]:


accuracy_score(pred,y_test)


# In[ ]:




