# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 09:35:06 2021

@author: kevin
"""

#############################################################
import tensorflow as tf
import matplotlib.pyplot as plt
import keras

tf.keras.layers.MaxPool2D(
    pool_size=(2, 2), strides=None, padding='valid', data_format=None,
    **kwargs
)

x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.],
                 [7., 8., 9.]])
x = tf.reshape(x, [1, 3, 3, 1])
max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
   strides=(1, 1), padding='valid')
max_pool_2d(x)


x = tf.constant([[1., 2., 3., 4.],
                 [5., 6., 7., 8.],
                 [9., 10., 11., 12.]])
x = tf.reshape(x, [1, 3, 4, 1])
max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
   strides=(1, 1), padding='valid')
max_pool_2d(x)



input_image = tf.constant([[[[1.], [1.], [2.], [4.]],
                           [[2.], [2.], [3.], [2.]],
                           [[4.], [1.], [1.], [1.]],
                           [[2.], [2.], [1.], [4.]]]])
output = tf.constant([[[[1], [0]],
                      [[0], [1]]]])
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
   input_shape=(4,4,1)))
model.compile('adam', 'mean_squared_error')
model.predict(input_image, steps=1)



x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.],
                 [7., 8., 9.]])
x = tf.reshape(x, [1, 3, 3, 1])
max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
   strides=(1, 1), padding='same')
max_pool_2d(x)