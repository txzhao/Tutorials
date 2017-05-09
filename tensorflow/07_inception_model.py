#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 17:41:43 2017

@author: txzhao
"""

import matplotlib.pyplot as plt
import matplotlib.image as mplImage
import os
import inception


# ================  define function  ================
def classify(image_path, showimage = False):
    if showimage:
        # display the image
        image = mplImage.imread(image_path)
        plt.figure()
        plt.axis('off')
        plt.imshow(image)
    
    # use the inception model to classify
    pred = model.classify(image_path = image_path)
    
    # print the scores and names of top-10 predictions
    model.print_scores(pred = pred, k = 10, only_first_name = True)
  

# plot the resized image which is fed into the model
def plot_resized_image(image_path):
    resized_image = model.get_resized_image(image_path = image_path)
    plt.figure()
    plt.axis('off')
    plt.imshow(resized_image, interpolation = 'nearest')
    plt.show()

    
# ================  run experiments  ================
# download the inception model and load it
inception.maybe_download()
model = inception.Inception()

# experiment 1
print("======= Experiment 1 =======")
print("- ground truth: cropped panda")
print("- scores: ")
image_path = os.path.join(inception.data_dir, 'cropped_panda.jpg')
classify(image_path, showimage = False)
print("")

# experiment 2
# note inception model works on images with 299x299 pixels
# input images with a different size will be resized automatically
print("======= Experiment 2 =======")
print("- ground truth: parrot")
print("- scores: ")
classify(image_path = "images/parrot.jpg", showimage = False)
plot_resized_image(image_path = "images/parrot.jpg")
print("")

# experiment 3
print("======= Experiment 3 =======")
print("- ground truth: cropped parrot top")
print("- scores: ")
classify(image_path = "images/parrot_cropped1.jpg", showimage = False)
print("")

# experiment 4
print("======= Experiment 4 =======")
print("- ground truth: cropped parrot middle")
print("- scores: ")
classify(image_path = "images/parrot_cropped2.jpg", showimage = False)
print("")

# experiment 5
print("======= Experiment 5 =======")
print("- ground truth: cropped parrot bottom")
print("- scores: ")
classify(image_path = "images/parrot_cropped3.jpg", showimage = False)
print("")

# experiment 6
print("======= Experiment 6 =======")
print("- ground truth: padded parrot")
print("- scores: ")
classify(image_path = "images/parrot_padded.jpg", showimage = False)
print("")

# experiment 7
print("======= Experiment 7 =======")
print("- ground truth: Elon Musk")
print("- scores: ")
classify(image_path = "images/elon_musk.jpg", showimage = False)
print("")

# experiment 8
print("======= Experiment 8 =======")
print("- ground truth: Elon Musk 100x100")
print("- scores: ")
classify(image_path = "images/elon_musk_100x100.jpg", showimage = False)
print("")

# experiment 9
print("======= Experiment 9 =======")
print("- ground truth: Willy Wonka old")
print("- scores: ")
classify(image_path = "images/willy_wonka_old.jpg", showimage = False)
print("")

# experiment 10
print("======= Experiment 10 =======")
print("- ground truth: Willy Wonka new")
print("- scores: ")
classify(image_path = "images/willy_wonka_new.jpg", showimage = False)
print("")
