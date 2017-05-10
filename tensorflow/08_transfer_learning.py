#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:52:16 2017

@author: txzhao
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
import inception
import prettytensor as pt
import cifar10
from cifar10 import num_classes
from inception import transfer_values_cache
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

# ================  define function  ================
def plot_images(images, cls_true, cls_pred = None, smooth = True):
	assert len(images) == len(cls_true)

	# create subplots
	fig, axes = plt.subplots(3, 3)

	# adjust vertical spacing
	if cls_pred is None:
		hspace = 0.3
	else:
		hspace = 0.6
	fig.subplots_adjust(hspace = hspace, wspace = 0.3)

	# interpolation option
	if smooth:
		interpolation = 'spline16'
	else:
		interpolation = 'nearest'

	for i, ax in enumerate(axes.flat):
		if i < len(images):
			# plot subplots
			ax.imshow(images[i], interpolation = interpolation)

			# print true and predicted names
			cls_true_name = class_names[cls_true[i]]
			if cls_pred is None:
				xlabel = "True: {0}".format(cls_true_name)
			else:
				cls_pred_name = class_names[cls_pred[i]]
				xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

			# set xlabel
			ax.set_xlabel(xlabel)

		ax.set_xticks([])
		ax.set_yticks([])

	plt.show()


def plot_transfer_values(i):
    print("input image:")
    
    # plot i'th image from the test-set
    plt.figure()
    plt.imshow(images_test[i], interpolation = 'nearest')
    plt.show()

    print("transfer values for the image using Inception model:")

    # transform the transfer-values into an image
    img = transfer_values_test[i]
    img = img.reshape((32, 64))

    # plot the image for the transfer-values
    plt.figure()
    plt.imshow(img, interpolation = 'nearest', cmap = 'Reds')
    plt.show()


def plot_scatter(values, cls):
	# create a color map
	cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))

	# get the color for each sample
	colors = cmap[cls]

	# extract x and y values
	x = values[:, 0]
	y = values[:, 1]

	plt.figure()
	plt.scatter(x, y, color = colors)
	plt.show()


# select a random batch from training samples
def random_batch(train_batch_size = 64):
    num_images = len(transfer_values_train)
    
    # create a random index and select samples based on it
    idx = np.random.choice(num_images, size = train_batch_size, replace = False)
    x_batch = transfer_values_train[idx]
    y_batch = labels_train[idx]
    
    return x_batch, y_batch


# mini-batch optimization
def optimize(num_iterations, train_batch_size = 64):   
    start_time = time.time()
    
    for i in range(num_iterations):
        # get a batch of training data
        x_batch, y_true_batch = random_batch(train_batch_size = train_batch_size)
        
        # put batch into a dict with the proper names
        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        # run the optimizer
        i_global, _ = session.run([global_step, optimizer], feed_dict = feed_dict_train)
        
        # print intermidiate results every 100 iterations
        if (i_global%100 == 0) or (i == num_iterations - 1):
            # calculate accuracy on training dataset and print
            batch_acc = session.run(accuracy, feed_dict = feed_dict_train)
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))
    
    # update time usage
    end_time = time.time()
    time_diff = end_time - start_time
    print("Time Usage: " + str(timedelta(seconds = int(round(time_diff)))))
  

# plot misclassified images
def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    
    # get misclassified images
    images = images_test[incorrect]
    
    # get corresponding predicted classes and ground truth
    cls_pred = cls_pred[incorrect]
    cls_true = cls_test[incorrect]
    
    n = min(9, len(images))
    plot_images(images = images[0:n], cls_true = cls_true[0:n], cls_pred = cls_pred[0:n])
    
    
# plot and print confusion matrix
def plot_confusion_matrix(cls_pred):
    # true classification
    cls_true = cls_test
    
    # get the confusion matrix and print out
    cm = confusion_matrix(y_true = cls_true, y_pred = cls_pred)
    # Print the confusion matrix as text.
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))
    

def predict_cls(transfer_values, labels, cls_true, batch_size = 256):
    # number of images
    num_images = len(transfer_values)

    # allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape = num_images, dtype = np.int)

    i = 0
    while i < num_images:
        # The ending index for the next batch is denoted j
        j = min(i + batch_size, num_images)
        feed_dict = {x: transfer_values[i:j], y_true: labels[i:j]}

        # Calculate the predicted class using TensorFlow
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict = feed_dict)

        i = j
        
    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred


def predict_cls_test(batch_size = 256):
    return predict_cls(transfer_values = transfer_values_test, labels = labels_test,
                       cls_true = cls_test, batch_size = batch_size)
    

def classification_accuracy(correct):
    return correct.mean(), correct.sum()


def print_test_accuracy(show_example_errors = False, show_confusion_matrix = False):
    # calculate the predicted classes and whether they are correct
    correct, cls_pred = predict_cls_test(batch_size = 256)
    
    # classification accuracy and the number of correct classifications
    acc, num_correct = classification_accuracy(correct)
    num_images = len(correct)

    # print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # plot some examples of mis-classifications
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred = cls_pred, correct = correct)

    # plot the confusion matrix
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred = cls_pred)


def main():
    num_iterations = 10000
    train_batch_size = 64
    optimize(num_iterations = num_iterations, train_batch_size = train_batch_size)
    print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)


# ================  data preparation  ================
# download the dataset 
cifar10.maybe_download_and_extract()

# load class-names, training data and test data
class_names = cifar10.load_class_names()
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

# print out the size of dataset
print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))

# plot a few images to see if data is right
images = images_test[0:9]
cls_true = cls_test[0:9]
plot_images(images = images, cls_true = cls_true, smooth = False)


# ================  inception model ================
# download and load the inception model
inception.maybe_download()
model = inception.Inception()

# set path for cache files
file_path_cache_train = os.path.join(cifar10.data_path, 'inception_cifar10_train.pkl')
file_path_cache_test = os.path.join(cifar10.data_path, 'inception_cifar10_test.pkl')

# scale images from [0,1] to [0,255]
images_train_scaled = images_train*255.0
images_test_scaled = images_test*255.0		

# calculate transfer-values and save them to a cache-file
print("Processing Inception transfer-values for training-images ...")
transfer_values_train = transfer_values_cache(cache_path = file_path_cache_train, 
                                              images = images_train_scaled, model = model)
print("Processing Inception transfer-values for test-images ...")
transfer_values_test = transfer_values_cache(cache_path = file_path_cache_test, 
                                              images = images_test_scaled, model = model)

plot_transfer_values(i = 16)


# ================  transfer-values analysis: PCA ================
# create a PCA object and set target array length to 2
pca = PCA(n_components = 2)
transfer_values = transfer_values_train[0:3000]
cls = cls_train[0:3000]

# reduce the transfer-values array from 2048 to 2
transfer_values_reduced = pca.fit_transform(transfer_values)
plot_scatter(values = transfer_values_reduced, cls = cls)


# ================  transfer-values analysis: t-SNE ================
# first reduce dimensionality to accelerate
pca = PCA(n_components = 50)
transfer_values_50d = pca.fit_transform(transfer_values)

# create a t-SNE object
tsne = TSNE(n_components = 2)
transfer_values_reduced = tsne.fit_transform(transfer_values_50d)
plot_scatter(values = transfer_values_reduced, cls = cls)


# ================  new TF classifier ================
# prepare placeholders
transfer_len = model.transfer_len
x = tf.placeholder(tf.float32, shape = [None, transfer_len], name = 'x')
y_true = tf.placeholder(tf.float32, shape = [None, num_classes], name = 'y_true')
y_true_cls = tf.argmax(y_true, dimension = 1)

# wrap the transfer-values as a PrettyTensor object
x_pretty = pt.wrap(x)

# create a fc classifier layer
with pt.defaults_scope(activation_fn = tf.nn.relu):
	y_pred, loss = x_pretty.\
        fully_connected(size = 1024, name = 'layer_fc1').\
        softmax_classifier(num_classes = num_classes, labels = y_true)

# optimization method
global_step = tf.Variable(initial_value = 0, name = 'global_step', trainable = False)
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(loss, global_step)

# classification accuracy
y_pred_cls = tf.argmax(y_pred, dimension = 1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ================  TensorFlow run  ================
session = tf.Session()
session.run(tf.global_variables_initializer())


main()
