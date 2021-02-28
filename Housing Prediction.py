# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 21:59:55 2021

@author: kevin
"""


import tensorflow as tf
import keras as kr
import numpy as np

# GRADED FUNCTION: house_model
def house_model(y_new):
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype = float)
    ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype = float)
    model = kr.Sequential([kr.layers.Dense(units = 1, input_shape = [1])])
    model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
    model.fit(xs, ys, epochs = 1000)
    return model.predict(y_new)[0]

house_model([7.0])