# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 20:24:43 2023

@author: Hamza
"""

import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
#lib for lazy
from lazypredict.Supervised import LazyClassifier
#from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
#-------

folder_path = 'C:/Users/Hamza/Desktop/dataset50'

folder_images = glob(folder_path + '/*/*.jpg')


#hog = cv2.HOGDescriptor()
from skimage.feature import hog
image_features = []
label_features=[]
total_images=len(folder_images)
for i,image_path in enumerate(folder_images):
    ir_=os.path.basename(os.path.dirname(image_path))
    image = cv2.imread(image_path)
    image1=image[...,2]
    hog = cv2.HOGDescriptor()
    #imagea=np.expand_dims(image1,-1)
    fd = hog.compute(image1)
    
    #plt.imshow(hog_image,cmap='gray')
    #break
    #features = hog.compute(image)
    image_features.append(fd)
    label_features.append(ir_)
    print(i+1, '/' , total_images,'-->',round((i+1)/total_images*100,4),'%')
X=np.array(image_features)
y=np.array(label_features)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)


clf = LazyClassifier(verbose=1,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)



#image = cv2.imread('a.jpg', cv2.IMREAD_GRAYSCALE)
#resized_image = cv2.resize(image, (64, 128))
# Create a HOG descriptor
#hog = cv2.HOGDescriptor()

# Compute the HOG features
#features = hog.compute(resized_image)


