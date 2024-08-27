# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 20:12:39 2023

@author: Hamza
"""

import cv2

# Load the image
image = cv2.imread('a.jpg', cv2.IMREAD_GRAYSCALE)
resized_image = cv2.resize(image, (64, 128))
# Create a HOG descriptor
hog = cv2.HOGDescriptor()

# Compute the HOG features
features = hog.compute(resized_image)