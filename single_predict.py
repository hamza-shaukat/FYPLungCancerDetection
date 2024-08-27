# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 23:21:19 2023

@author: Hamza
"""



import pickle
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
from sklearn.metrics import confusion_matrix


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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
import datetime

from skimage.feature import hog



# Load the image
img = cv2.imread('C:/Users/Hamza/Desktop/work/a.jpg')

# Resize the image
img = cv2.resize(img, (256, 256))

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#%%
# Reshape the image to 2D array
gray=np.expand_dims(gray, axis=2)

hog_features = []
#%%

hog_features = hog(gray, channel_axis=2)

#%%

X = np.array(hog_features)
#%%

with open('C:/Users/Hamza/Desktop/project/Machine_learning/saved_models/lda.pkl', 'rb') as file:
    model = pickle.load(file)

#%%
prediction = model.predict(X.reshape(1, -1))
#print(prediction)
if(prediction=='0'):
    print("Negative")
elif(prediction=='1'):
    print("Positive")































