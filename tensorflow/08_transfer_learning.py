#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:52:16 2017

@author: txzhao
"""

import matplotlib.pyplot as plt
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
	plt.imshow(images_test[i], interpolation = 'nearest')
	plt.show()

	print("transfer values for the image using Inception model:")

	# transform the transfer-values into an image
	img = transfer_values_test[i]
	img = img.reshape((32, 64))

	# plot the image for the transfer-values
	plt.imshow(img, interpolation = 'nearest', cmap = 'Reds')
	plt.show()


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


# ================  inception model setup ================
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


