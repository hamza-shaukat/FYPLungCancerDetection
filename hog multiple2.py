# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 23:15:55 2022

@author: Hamza
"""

import cv2
import numpy as np

from glob import glob

#lib for lazy
from lazypredict.Supervised import LazyClassifier
#from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
#-------

folder_path = 'C:/Users/Hamza/Desktop/try'

folder_images = glob(folder_path + '/*/*.jpg')


#hog = cv2.HOGDescriptor()
from skimage.feature import hog
image_features = []
label_features=[]

for image_path in folder_images:
    
    image = cv2.imread(image_path)
    image1=image[...,2]
    #imagea=np.expand_dims(image1,-1)
    fd, hog_image = hog(image1, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=False)
    #features = hog.compute(image)
    image_features.append(fd)
    
x=np.array(image_features)


folder2_features = []
for image_path in folder2_images:
    image = cv2.imread(image_path)
    features = hog.compute(image)
    folder2_features.append(features)
    
#data = folder1_features.append(features)
X = folder1_features
y= folder2_features

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)