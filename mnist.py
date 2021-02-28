# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 19:09:01 2021

@author: kevin
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import keras
path = 'C:\\Users\\kevin\\.keras\\datasets\\mnist.npz'


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data(path=path)

x_train = x_train / 255.0
x_test = x_test / 255.0


print(y_train[0])
print(y_test[0])

plt.imshow(x_train[0])
plt.imshow(x_test[0])

plt.imshow(y_train[0])
plt.imshow(y_test[0])


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), ## shape and size of data
    tf.keras.layers.Dense(128, activation = tf.nn.relu), ## 128 neurons, like variables
    #tf.keras.layers.Dense(64, activation = tf.nn.relu), ## 128 neurons, like variables
    tf.keras.layers.Dense(10, activation = tf.nn.softmax) ## 10 classes of clothes in data set
    ])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 10, callbacks=[callbacks])

model.evaluate(x_test, y_test)

classifications = model.predict(x_test)
print(classifications[0])
print(y_test[0])


plt.imshow(y_test[0])