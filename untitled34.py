# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 23:05:28 2022

@author: Hamza
"""

import glob
from skimage import feature
import cv2

def extract_hog_features(images):
  hog_features = []
  for image in images:
    hog_features.append(feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=False))
  return hog_features

# List all the images in the first folder
image_files_1 = glob.glob("C:/Users/Hamza/Desktop/aaaa/*.jpg")

# Load the images from the first folder
images_1 = []
for file in image_files_1:
  image = cv2.imread(file)
  images_1.append(image)

# Extract HOG features for the images from the first folder
hog_features_1 = extract_hog_features(images_1)

# List all the images in the second folder
image_files_2 = glob.glob("C:/Users/Hamza/Desktop/try/1*.jpg")

# Load the images from the second folder
images_2 = []
for file in image_files_2:
  image = cv2.imread(file)
  images_2.append(image)
  
hog_features_2 = extract_hog_features(images_2)