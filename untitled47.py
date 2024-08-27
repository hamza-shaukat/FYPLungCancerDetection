# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 19:22:45 2023

@author: Hamza
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 19:10:21 2023

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from sklearn.linear_model import RidgeClassifier
#from sklearn.externals import joblib
import pickle
from skimage.feature import hog
#-------

#folder_path = 'C:/Users/Hamza/Desktop/dataset50gs'

#folder_images = glob(folder_path + '/*/*.jpg')

image_features = []


with open('ridge.pkl', 'rb') as file:
    model=pickle.load(file)
    
    
img=cv2.imread('C:/Users/Hamza/Desktop/predict/1/000b1d3f6ad8059dc5b3c3e4cbd93877.jpg')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gs1=np.expand_dims(gray, axis=2)

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gs1=np.expand_dims(gray, axis=2)
    #print(type(gray),gray.shape)
#image_array=np.expand_dims(gray, axis=2)
    #print(type(gs1),gs1.shape)
    #break
    # Save the grayscale image
#image_array=np.expand_dims(image_array, axis=0)
#image_array=image_array.reshape(1, 512,512,1)

def extract_features(gs1):
    
    #image1=img[...,2]
    fd, hog_image = hog(gs1, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    return fd
#image_features.append(fd)

prediction=model.predict(extract_features(img))
print(prediction)
