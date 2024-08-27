# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 20:09:12 2023

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

from skimage.feature import hog
import pickle
#-------


with open('rf.pkl', 'rb') as file:
    model=pickle.load(file)


folder_path = 'C:/Users/Hamza/Desktop/predict0'

folder_images = glob(folder_path + '/*/*.jpg')


#hog = cv2.HOGDescriptor()

image_features = []
label_features=[]
total_images=len(folder_images)
for i,image_path in enumerate(folder_images):
    ir_=os.path.basename(os.path.dirname(image_path))
    image = cv2.imread(image_path)
    image1=image[...,2]
    #imagea=np.expand_dims(image1,-1)
    fd, hog_image = hog(image1, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=False)
    
    #plt.imshow(hog_image,cmap='gray')
    #break
    #features = hog.compute(image)
    image_features.append(fd)
#    label_features.append(ir_)
#    print(i+1, '/' , total_images,'-->',round((i+1)/total_images*100,4),'%')
#X=np.array(image_features)
#y=np.array(label_features)


prediction=model.predict(image_features)
print(prediction)