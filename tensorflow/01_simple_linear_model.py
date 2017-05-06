#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:52:26 2017

@author: MMGF2
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
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
      
        
# mini-batch SGD
def optimize(num_iterations, batch_size):
    for i in range(num_iterations):
        # get a batch of raining examples
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph
        # Note that the placeholder for y_true_cls is not set
        # because it is not used during training
        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        # Run the optimizer using this batch of training data
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer
        session.run(optimizer, feed_dict = feed_dict_train)
        
 
# print test accuracy
def print_accuracy():
    # compute and output test accuracy 
    acc = session.run(accuracy, feed_dict = feed_dict_test)
    print("Accuracy on test-set: {0:.1%}".format(acc))
            

# plot and print confusion matrix
def print_confusion_matrix():
    # Get the true classifications for the test-set
    cls_true = data.test.cls
    
    # Get the predicted classifications for the test-set
    cls_pred = session.run(y_pred_cls, feed_dict = feed_dict_test)

    # Get the confusion matrix using sklearn
    cm = confusion_matrix(y_true = cls_true, y_pred = cls_pred)
    
    # Print the confusion matrix as text
    print(cm)
    
    # Plot the confusion matrix as an image
    plt.figure()
    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
   
    # Make various adjustments to the plot
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    
# plot misclassified examples of images
def plot_example_errors():
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image
    correct, cls_pred = session.run([correct_prediction, y_pred_cls], feed_dict = feed_dict_test)
    
    # pick out misclassified samples
    incorrect = (correct == False)
    
    # fetch images classified incorrectly
    images = data.test.images[incorrect]
    
    # get the predicted and true classes
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]

    # plot the first 9 images
    plot_images(images = images[0:9], cls_true = cls_true[0:9], cls_pred = cls_pred[0:9])
       

# plot the weights of model
def plot_weights():
    # Get the values for the weights from the TensorFlow variable
    w = session.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)
    
    # create sub-plots
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots
        if i < 10:
            # Get the weights for the i'th digit and reshape it
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot
            ax.set_xlabel("Weights: {0}".format(i))

            # plot the image
            ax.imshow(image, vmin = w_min, vmax = w_max, cmap = 'seismic')

        # remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        
def main(num_iterations, batch_size):
    optimize(num_iterations = num_iterations, batch_size = batch_size)
    print_accuracy()
    plot_example_errors()
    plot_weights()
    print_confusion_matrix()
    
    
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
data.test.cls = np.array([label.argmax() for label in data.test.labels])

# obtain data dimensions
img_size_flat = np.shape(data.train.images)[1]      # image size in the form of a 1d vector
img_size = np.sqrt(img_size_flat)                   # image size in each dimension
img_shape = (img_size, img_size)                    # tuple with height and width of images
num_classes = np.shape(data.test.labels)[1]         # number of classes

# get images from the test-set
images = data.test.images[0:9]

# get true classes for these images
cls_true = data.test.cls[0:9]

# plot images and labels calling function plot_images
plot_images(images = images, cls_true = cls_true)


## ==============  tensorflow graph setup  ================
# define placeholder for input images, true labels and true classes
x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

# define variables - weights and biases
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))

# create a mathematical model
logits = tf.matmul(x, weights) + biases
y_pred = tf.nn.softmax(logits)                      # normalization with softmax
y_pred_cls = tf.arg_max(y_pred, dimension = 1)      # pick out the largest element as the predicted class   

# calculate cross-entropy loss
# use logits instead of y_pred due to an internal softmax
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_true)
cost = tf.reduce_mean(cross_entropy)                # take the average of the cross_entropy 

# create an optimizer (grdient descent)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5).minimize(cost)

# calculate accuracy
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


## ==============  tensorflow run  ================
# create a tensorflow session
session = tf.Session()

# initialize variables
session.run(tf.global_variables_initializer())

# define dictionary with test data
feed_dict_test = {x: data.test.images, y_true: data.test.labels, y_true_cls: data.test.cls}


main(num_iterations = 1000, batch_size = 1000)

    



    