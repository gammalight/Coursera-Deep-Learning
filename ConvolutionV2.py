# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 09:06:07 2021

@author: kevin
"""

# conda install -c menpo opencv
import cv2
from scipy import misc
i = misc.ascent()

import matplotlib.pyplot as plt
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()

import numpy as np
from tensorflow import keras
import keras

import tensorflow as tf
print(tf.__version__)

i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

###################################################################################
# this filter detects edges nicely
# it creates a convolution that only passes through sharp edges and straight lines

# Experiment with different values for fun effects
#filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]

filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]] # vertical
# filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] # horizontal

# If all the digits in the filter dont add up to 0 or 1, you
# should probably do a weight to get it to do so
# so, for exampole, if your weights are 1,1,1 1,1,2, 1,1,1
# they add up to 10, so you would set a weight of 0.1 if you want to normalize them
###################################################################################

weight = 1

for x in range(1, size_x - 1):
    for y in range(1, size_y - 1):
        convolution = 0.0
        convolution = convolution + (i[x - 1, y - 1] * filter[0][0])
        convolution = convolution + (i[x, y - 1] * filter[0][1])
        convolution = convolution + (i[x + 1, y - 1] * filter[0][2])
        convolution = convolution + (i[x - 1, y] * filter[1][0])
        convolution = convolution + (i[x, y] * filter[1][1])
        convolution = convolution + (i[x + 1, y] * filter[1][2])
        convolution = convolution + (i[x - 1, y + 1] * filter[2][0])
        convolution = convolution + (i[x, y + 1] * filter[2][1])
        convolution = convolution + (i[x + 1, y + 1] * filter[2][2])
        convolution = convolution * weight
        if(convolution < 0):
            convolution = 0
        if(convolution > 255):
            convolution = 255
        i_transformed[x, y] = convolution
            

plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
plt.show()


# pooling

new_x = int(size_x / 2)
new_y = int(size_y / 2)

newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
        pixels = []
        pixels.append(i_transformed[x, y])
        pixels.append(i_transformed[x, y])
        pixels.append(i_transformed[x, y])
        pixels.append(i_transformed[x, y])
        pixels.sort(reverse = True)
        newImage[int(x / 2), int(y / 2)] = pixels[0]

plt.gray()
plt.grid(False)
plt.imshow(newImage)
#plt.axis('off')
plt.show()    

