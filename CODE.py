# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:46:47 2020

@author: Navya Kommareddy
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df =pd.read_csv('dataset/CSV_datasetsix_vowel_dataset_with_class.csv')
df.head()

pix=[]
for i in range(784):
    pix.append('pixel'+str(i))
features=pix
X = df.loc[:, features].values
y = df.loc[:,'class'].values

X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size = 0.30, random_state = 100)
y_train=y_train.ravel()
y_test=y_test.ravel()

def row2img(data):
    return np.asfarray(data).reshape((28,28))


data=X_train[11]
f, ax1 = plt.subplots(1, 1, sharey=True)
f.suptitle('Respective image of X_train[11]', size='20')
ax1.imshow(255-row2img(data), cmap=plt.cm.binary);    

X_train = tf.keras.utils.normalize(X_train,axis=1)
X_test = tf.keras.utils.normalize(X_test,axis=1)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(7,activation=tf.nn.softmax))

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=50)


_,acc=model.evaluate(X_test,y_test)
print('Accuracy: {}'.format(acc))

pred=model.predict([X_test])
print('Predicted Label: ',np.argmax(pred[11]))

f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1.set_title('Actual Label: '+str(y_test[11]))
ax1.imshow(255-row2img(X_test[11]),cmap=plt.cm.binary);