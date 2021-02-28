# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:01:45 2021

@author: kevin
"""

import numpy as np
from tensorflow import keras

import tensorflow as tf
print(tf.__version__)
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0


print(train_labels[0])
print(train_images[0])

plt.imshow(train_images[0])


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.98):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()



model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), ## shape and size of data
    tf.keras.layers.Dense(512, activation = tf.nn.relu), ## 128 neurons, like variables
    tf.keras.layers.Dense(10, activation = tf.nn.softmax) ## 10 classes of clothes in data set
    ])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs = 10, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

