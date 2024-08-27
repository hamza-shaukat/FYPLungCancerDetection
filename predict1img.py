# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 17:51:50 2023

@author: Hamza
"""

import cv2
import numpy as np
from skimage.feature import hog
import pickle

# Load the image
img = cv2.imread('C:/Users/Hamza/Desktop/dataset5k/train/0/000a27d90292ae38efb09d0ce84ab7d3.jpg')

# Resize the image
img = cv2.resize(img, (256, 256))

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#%%
# Reshape the image to 2D array
gray=np.expand_dims(gray, axis=2)

hog_features = []
#%%
# Extract HOG features from the image
hog_features = hog(gray, channel_axis=2)
#hog_features=hog_features.append(hog(gray, channel_axis=2))
#%%
# Convert the HOG features to a numpy array
X = np.array(hog_features)
#%%
# Load the saved model
with open('C:/Users/Hamza/Desktop/project/Machine_learning/saved_models/lda.pkl', 'rb') as file:
    model = pickle.load(file)

# Make a prediction on the image
prediction = model.predict(X.reshape(1, -1))
print(prediction)
