# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 01:20:09 2023

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

import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog


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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
import datetime

import numpy as np
#z=np.load('C:/Users/Hamza/Desktop/dtsavedfiles/label5k.npy'


import numpy as np
from skimage.feature import hog

# Load the numpy file containing the images
images = np.load('C:/Users/Hamza/Desktop/saved_dataset/image5k.npy')

# Initialize an empty list to store the HOG features
hog_features = []

# Iterate through the images
for image in images:
    # Apply HOG to the current image and append the resulting feature vector to the list
    hog_features.append(hog(image, channel_axis=2))
    #fd, hog_image = hog(image, visualise = True)
    
X=np.array(hog_features)
y=np.load('C:/Users/Hamza/Desktop/saved_dataset/label5k.npy')





































X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)

# Random Forest
rf = RandomForestClassifier()
# Cross-validation
scores = cross_val_score(rf, X_train, y_train, cv=5)
print("Cross-validation scores for Random Forest:", scores)
# Grid Search
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10, 15]}
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters for Random Forest:", grid_search.best_params_)
rf = grid_search.best_estimator_
rf.fit(X_train, y_train)
accuracy = rf.score(X_test, y_test)
print("Accuracy of Random Forest: {:.2f}%".format(accuracy * 100))
with open('rf.pkl', 'wb') as file:
    pickle.dump(rf, file)

# LDA
lda = LinearDiscriminantAnalysis()
# Cross-validation
scores = cross_val_score(lda, X_train, y_train, cv=5)
print("Cross-validation scores for LDA:", scores)
lda.fit(X_train, y_train)
accuracy = lda.score(X_test, y_test)
print("Accuracy of LDA: {:.2f}%".format(accuracy * 100))
#with open:
    
# LDA
with open('lda.pkl', 'wb') as file:
    pickle.dump(lda, file)
    
# SVC
sv = svm.SVC(kernel='sigmoid')
# Cross-validation
scores = cross_val_score(sv, X_train, y_train, cv=5)
print("Cross-validation scores for SVC:", scores)
# Grid Search
param_grid = {'C': [0.1, 1, 10], 'kernel': ['sigmoid', 'linear']}
grid_search = GridSearchCV(sv, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters for SVC:", grid_search.best_params_)
sv = grid_search.best_estimator_
sv.fit(X_train, y_train)
accuracy = sv.score(X_test, y_test)
print("Accuracy of SVC: {:.2f}%".format(accuracy * 100))
with open('svc.pkl', 'wb') as file:
    pickle.dump(sv, file)

# load the saved model
loaded_model_rf = load_model('rf.pkl')
loaded_model_lda = load_model('lda.pkl')
loaded_model_svc = load_model('svc.pkl')

# predict on a new image
new_image = cv2.imread('path/to/new/image.jpg')
new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
prediction_rf = predict(loaded_model_rf, new_image)
prediction_lda = predict(loaded_model_lda, new_image)
prediction_svc = predict(loaded_model_svc, new_image)

print("Prediction using Random Forest:", prediction_rf)
print("Prediction using LDA:", prediction_lda)
print("Prediction using SVC:", prediction_svc)

# generating confusion matrix and metrics
y_true = y_test
y_pred_rf = loaded_model_rf.predict(X_test)
y_pred_lda = loaded_model_lda.predict(X_test)
y_pred_svc = loaded_model_svc.predict(X_test)

cm_rf = confusion_matrix(y_true, y_pred_rf)
print("Confusion Matrix for Random Forest:", cm_rf)
metrics_rf = classification_report(y_true, y_pred_rf)
print("Metrics for Random Forest:", metrics_rf)

cm_lda = confusion_matrix(y_true, y_pred_lda)
print("Confusion Matrix for LDA:", cm_lda)
metrics_lda = classification_report(y_true, y_pred_lda)
print("Metrics for LDA:", metrics_lda)

cm_svc = confusion_matrix(y_true, y_pred_svc)
print("Confusion Matrix for SVC:", cm_svc)
metrics_svc = classification_report(y_true, y_pred_svc)
print("Metrics for SVC:", metrics_svc)