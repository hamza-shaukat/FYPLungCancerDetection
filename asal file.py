# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 12:29:54 2023

@author: Hamza
"""

#Here is an example of how to use the glob module to get a list of all the image files in a folder and then follow the same steps as before to get the dimensions of each image and reshape it to (10000,1):

#Copy code
import glob
import cv2
import numpy as np
import os

# folder containing the images
folder = "C:/Users/Hamza/Desktop/dataset50gs/0"

# use glob to get a list of all the image files in the folder
image_files = glob.glob(os.path.join(folder, "*.jpg"))

# create an empty list to store the image dimensions
image_dims = []

# loop through the images and get their dimensions
for image_file in image_files:
    # open the image
    img = cv2.imread(image_file)
    # get the dimensions of the image
    h, w, _ = img.shape
    # append the dimensions to the list
    image_dims.append((h, w))

# convert the list of dimensions to a numpy array
image_dims_array = np.array(image_dims)

# reshape the images to (10000, 1)
img = img.reshape(10000,1)