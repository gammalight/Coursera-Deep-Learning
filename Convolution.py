# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 09:15:12 2021

@author: kevin
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3, 3))
y = model(x)

#############################################################
import tensorflow as tf
import matplotlib.pyplot as plt
import keras


tf.keras.layers.Conv2D(
    filters, kernel_size, strides=(1, 1), padding='valid',
    data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
    use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, **kwargs
)

# The inputs are 28x28 RGB images with `channels_last` and the batch
# size is 4.
input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D(
2, 3, activation='relu', input_shape=input_shape[1:])(x)
print(y.shape)

# With `dilation_rate` as 2.
input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D(
2, 3, activation='relu', dilation_rate=2, input_shape=input_shape[1:])(x)
print(y.shape)

# With `padding` as "same".
input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D(
3, 4, activation='relu', padding="same", input_shape=input_shape[1:])(x)
print(y.shape)

# With extended batch shape [4, 7]:
input_shape = (4, 7, 28, 28, 3)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D(
3, 4, activation='relu', padding="same", input_shape=input_shape[0:])(x)
print(y.shape)

##############################################################
##############################################################


import tensorflow as tf
import matplotlib.pyplot as plt
import keras
path = 'C:\\Users\\kevin\\.keras\\datasets\\mnist.npz'

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


train_images = train_images / 255.0
test_images = test_images / 255.0

print(train_labels[10])
print(train_images[10])

plt.imshow(train_images[10])


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.98):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()


model = tf.keras.models.Sequential([
#tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (28, 28, 1)), 
#64 filters, 3x3 size, input shape 28x28, single color depth
#tf.keras.layers.MaxPooling2D(2, 2),
#tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
#tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Flatten(), ## shape and size of data
tf.keras.layers.Dense(512, activation = tf.nn.relu), ## 128 neurons, like variables
tf.keras.layers.Dense(10, activation = tf.nn.softmax) ## 10 classes of clothes in data set
    ])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs = 10, callbacks = [callbacks])

model.summary()

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

plt.imshow(test_labels[0])



##################################################################


import numpy as np
from tensorflow import keras
import keras

import tensorflow as tf
print(tf.__version__)
import matplotlib.pyplot as plt


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.98):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()



path = 'C:\\Users\\kevin\\.keras\\datasets\\mnist.npz'

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)), 
#64 filters, 3x3 size, input shape 28x28, single color depth
tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Flatten(), ## shape and size of data
tf.keras.layers.Dense(128, activation = tf.nn.relu), ## 128 neurons, like variables
tf.keras.layers.Dense(10, activation = tf.nn.softmax) ## 10 classes of clothes in data set
    ])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs = 10, callbacks = [callbacks])

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_loss)

model.summary()


classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

plt.imshow(test_images[0])


##################################################################

print(test_labels[:100])

for i in [2, 3, 5, 15, 24]:
    print(i)
    plt.imshow(test_images[i])
    plt.show()

import matplotlib.pyplot as plt
f, axarr = plt.subplots(3, 4)
FIRST_IMAGE = 2
SECOND_IMAGE = 3
THIRD_IMAGE = 5
CONVOLUTION_NUMBER = 5
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
    f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap = 'inferno')
    axarr[0, x].grid(False)
    
    f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap = 'inferno')
    axarr[1, x].grid(False)
    
    f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap = 'inferno')
    axarr[2, x].grid(False)
    
    
    
    
    