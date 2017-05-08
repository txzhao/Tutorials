#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 15:27:49 2017

@author: txzhao
"""

import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np 
from sklearn.metrics import confusion_matrix
import time
import math
from datetime import timedelta
from tensorflow.examples.tutorials.mnist import input_data


## ==============  define functions  ================
# plot a single image
def plot_image(image):
    plt.figure()
    plt.imshow(image.reshape(img_shape), 
               interpolation = 'nearest', cmap = 'binary')
    plt.show()


# create a figure with a grid of subplots
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

# mini-batch optimization
def optimize(num_iterations, train_batch_size = 64):
    global total_iterations    
    start_time = time.time()
    
    for i in range(total_iterations, total_iterations + num_iterations):
        # get a batch of training data
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        
        # put batch into a dict with the proper names
        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        # run the optimizer
        session.run(optimizer, feed_dict = feed_dict_train)
        
        # print intermidiate results every 100 iterations
        if i%100 == 0:
            # calculate accuracy on training dataset and print
            acc = session.run(accuracy, feed_dict = feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))
    
    # update iterations and time usage
    total_iterations += num_iterations
    end_time = time.time()
    time_diff = end_time - start_time
    print("Time Usage: " + str(timedelta(seconds = int(round(time_diff)))))


# plot misclassified images
def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    
    # get misclassified images
    images = data.test.images[incorrect]
    
    # get corresponding predicted classes and ground truth
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    
    plot_images(images = images[0:9], cls_true = cls_true[0:9], cls_pred = cls_pred[0:9])
    

# plot and print confusion matrix
def plot_confusion_matrix(cls_pred):
    # true classification
    cls_true = data.test.cls
    
    # get the confusion matrix and print out
    cm = confusion_matrix(y_true = cls_true, y_pred = cls_pred)
    print(cm)
    plt.matshow(cm)
    
    # plot adjustment
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.show()
    

# evaluate the network with test data
def print_test_accuracy(test_batch_size = 256, show_example_errors = False, 
                        show_confusion_matrix = False):
    # number of images in the test set
    num_test = len(data.test.images)
    cls_pred = np.zeros(shape = num_test, dtype = np.int)
    
    i = 0
    while i < num_test:
        j = min(i + test_batch_size, num_test)      # ending index
        
        # get image batch and labels from i to j
        images = data.test.images[i:j, :]
        labels = data.test.labels[i:j, :]
        feed_dict = {x: images, y_true: labels}
        
        # calculate the predictd class
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict = feed_dict)
        i = j
    
    # calculate test accuracy and print out
    cls_true = data.test.cls
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum)/num_test
    msg = "Accuracy on Test-set: {0:.1%} ({1}/{2})"
    print(msg.format(acc, correct_sum, num_test))
    
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred = cls_pred, correct = correct)
    
    if show_confusion_matrix:
        print("Confusion matrix:")
        plot_confusion_matrix(cls_pred = cls_pred)   


# function for plotting convolutional weights
def plot_conv_weights(weights, input_channel = 0):
    # retrieve weights-variable from TensorFlow
    w = session.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)
    
    # obtain number of filters and number of grids for plotting
    num_filters = w.shape[3]
    num_grids = int(math.ceil(math.sqrt(num_filters)))
    
    # create figure with a grid of sub-plots
    fig, axes = plt.subplots(num_grids, num_grids)
    
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = w[:, :, input_channel, i]
            ax.imshow(img, vmin = w_min, vmax = w_max, 
                      interpolation = 'nearest', cmap = 'seismic')
        
        # remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()
    

# plot the output of a conv-layer
def plot_conv_layer(layer, image):
    # create a feed-dict for a single image
    feed_dict = {x: [image]}
    
    # feed network graph with one image and retrieve results
    values = session.run(layer, feed_dict = feed_dict)
    
    # obtain number of filters and number of grids for plotting
    num_filters = values.shape[3]
    num_grids = int(math.ceil(math.sqrt(num_filters)))
    
    # create figure with a grid of sub-plots
    fig, axes = plt.subplots(num_grids, num_grids)
    
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation = 'nearest', cmap = 'binary')
        
        # remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()
    

def main(num_iterations, train_batch_size, test_batch_size):  
    # train and output results
    optimize(num_iterations = num_iterations, train_batch_size = train_batch_size)
    print_test_accuracy(test_batch_size = test_batch_size, 
                        show_example_errors = True, show_confusion_matrix = True)
    
    # look into inner workings
    im_1 = data.test.images[0]
    plot_image(im_1)
    plot_conv_weights(weights = weights_conv_1)
    plot_conv_weights(weights = weights_conv_2)
    plot_conv_layer(layer = layer_conv_1, image = im_1)
    plot_conv_layer(layer = layer_conv_2, image = im_1)
    

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
img_size_flat = int(np.shape(data.train.images)[1])     # image size in the form of a 1d vector
img_size = int(np.sqrt(img_size_flat))                  # image size in each dimension
img_shape = (img_size, img_size)                        # tuple with height and width of images
num_classes = np.shape(data.test.labels)[1]             # number of classes
num_channels = 1                                        # number of color channels

# get images from the test-set
images = data.test.images[0:9]

# get true classes for these images
cls_true = data.test.cls[0:9]

# plot images and labels calling function plot_images
plot_images(images = images, cls_true = cls_true)


## ==============  tensorflow graph setup  ================
# network configuration
# ConvNet layer1
filter_size_1 = 5
num_filters_1 = 16

# ConvNet layer2
filter_size_2 = 5
num_filters_2 = 36

# fully-connected layer
fc_size = 128

# define placeholder variables
x = tf.placeholder(tf.float32, shape = [None, img_size_flat], name = 'x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape = [None, num_classes], name = 'y_true')
y_true_cls = tf.argmax(y_true, dimension = 1)

# covolutional layer1
layer_conv_1, weights_conv_1 = new_conv_layer(input = x_image, num_input_channels = num_channels, 
                                            filter_size = filter_size_1, num_filters = num_filters_1, 
                                            use_pooling = True)

# covolutional layer2
layer_conv_2, weights_conv_2 = new_conv_layer(input = layer_conv_1, num_input_channels = num_filters_1, 
                                            filter_size = filter_size_2, num_filters = num_filters_2, 
                                            use_pooling = True)

# flatten layer
layer_flat, num_features = flatten_layer(layer_conv_2)

# fully-connected layer1
layer_fc_1 = new_fc_layer(input = layer_flat, num_inputs = num_features, 
                          num_outputs = fc_size, use_relu = True)

# fully-connected layer2
layer_fc_2 = new_fc_layer(input = layer_fc_1, num_inputs = fc_size, 
                          num_outputs = num_classes, use_relu = False)

# predict classes
y_pred = tf.nn.softmax(layer_fc_2)
y_pred_cls = tf.argmax(y_pred, dimension = 1)

# calculate cost function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = layer_fc_2, labels = y_true)
cost = tf.reduce_mean(cross_entropy)

# optimization method
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cost)

# performance measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


## ==============  tensorflow run  ================
# create tensorflow session
session = tf.Session()

# initialize variables
session.run(tf.global_variables_initializer())

# initialize parameters
total_iterations = 0
num_iterations = 1000
train_batch_size = 64
test_batch_size = 256


main(num_iterations = num_iterations, train_batch_size = train_batch_size, 
     test_batch_size = test_batch_size)

