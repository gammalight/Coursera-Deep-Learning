# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:19:42 2021

@author: kevin
"""

import tensorflow as tf
import keras as kr
import numpy as np


model = kr.Sequential([kr.layers.Dense(units = 1, input_shape = [1])])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype = float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype = float)

model.fit(xs, ys, epochs = 500)
print(model.predict([10.0]))


# GRADED FUNCTION: house_model
def house_model(y_new):
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype = float)
    ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5], dtype = float)
    model = kr.Sequential([kr.layers.Dense(units = 1, input_shape = [1])])
    model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
    model.fit(xs, ys, epochs = 1000)
    return model.predict(y_new)[0]

house_model([12.0])



