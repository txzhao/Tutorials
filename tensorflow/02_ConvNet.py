#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 15:27:49 2017

@author: MMGF2
"""

import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np 
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from tensorflow.examples.tutorials.mnist import input_data


## ==============  define functions  ================
def plot_images(images, cls_true, cls_pred = None):
    # test the condition below and trigger an error if false
    assert len(images) == len(cls_true) == 9
    
    # create figure with 3X3 sub-plots
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
    
    for i, ax in enumerate(axes.flat):
        # plot image
        ax.imshow(images[i].reshape(img_shape), cmap = 'binary')
        
        # show true and predicted classes
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        
        ax.set_xlabel(xlabel)
        
        # remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()
  

# create weights variable
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))


# create biases variable
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape = [length]))    


# create convolutional layer
def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling = True):
    # shape of the filter-weights for the convolution (defined by tensorflow API)
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # create new weights
    weights = new_weights(shape = shape)
    
    # create new biases, one for each filter
    biases = new_biases(length = num_filters)
    
    # tensorflow operation for convolution
    layer = tf.nn.conv2d(input = input, filter = weights, 
                         strides = [1, 1, 1, 1], padding = 'SAME')      # strides = [batch, height, width, channels]
    layer += biases
    
    # use pooling to down-sample images
    if use_pooling:
        layer = tf.nn.max_pool(value = layer, ksize = [1, 2, 2, 1], 
                               strides = [1, 2, 2, 1], padding = 'SAME')
    
    # nonlinear activation ReLu
    layer = tf.nn.relu(layer)
    
    return layer, weights

# reduce the 4d tensor to 2d matrix
def flatten_layer(layer):
    layer_shape = layer.get_shape()         # layer_shape = [num_images, img_height, img_width, num_channels]
    
    # number of feature: img_height*img_width*num_channels
    num_features = layer_shape[1:4].num_elements()
    
    # reshape the layer to [num_images, num_features]
    # -1 means size in that dimension is indirectly computed
    layer_flat = tf.reshape(layer, [-1, num_features])      
    return layer_flat, num_features


# create fully-connected layer
def new_fc_layer(input, num_inputs, num_outputs, use_relu = True):
    # create new weights and biases
    weights = new_weights(shape = [num_inputs, num_outputs])
    biases = new_biases(length = num_outputs)
    
    # create a linear model
    layer = tf.matmul(input, weights) + biases
    
    # nonlinear activation ReLu
    if use_relu:
        layer = tf.nn.relu(layer)
    
    return layer


## ==============  network configuration  ================
# ConvNet layer1
filter_size_1 = 5
num_filters_1 = 16

# ConvNet layer2
filter_size_2 = 5
num_filters_2 = 36

# fully-connected layer
fc_size = 128


## ==============  data preparation  ================    
# load data automatically
data = input_data.read_data_sets("data/MNIST/", one_hot = True)

# an overview of data
print(" ")
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t\t{}".format(len(data.validation.labels)))

# convert one-hot presentation into class index
data.test.cls = np.argmax(data.test.labels, axis = 1)

# obtain data dimensions
img_size_flat = np.shape(data.train.images)[1]      # image size in the form of a 1d vector
img_size = np.sqrt(img_size_flat)                   # image size in each dimension
img_shape = (img_size, img_size)                    # tuple with height and width of images
num_classes = np.shape(data.test.labels)[1]         # number of classes
num_channels = 1                                    # number of color channels

# get images from the test-set
images = data.test.images[0:9]

# get true classes for these images
cls_true = data.test.cls[0:9]

# plot images and labels calling function plot_images
plot_images(images = images, cls_true = cls_true)


## ==============  tensorflow graph setup  ================
# define placeholder variables
x = tf.placeholder(tf.float32, shape = [None, img_size_flat], name = 'x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape = [None, num_classes], name = 'y_true')
y_true_cls = tf.argmax(y_true, dimension = 1)

# covolutional layer1
layer_conv1, weights_conv1 = new_conv_layer(input = x_image, num_input_channels = num_channels, 
                                            filter_size = filter_size_1, num_filters = num_filters_1, 
                                            use_pooling = True)

# covolutional layer2
layer_conv2, weights_conv2 = new_conv_layer(input = x_image, num_input_channels = num_channels, 
                                            filter_size = filter_size_1, num_filters = num_filters_1, 
                                            use_pooling = True)




