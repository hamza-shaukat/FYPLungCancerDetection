# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 12:39:46 2023

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



import numpy as np
#z=np.load('C:/Users/Hamza/Desktop/dtsavedfiles/label5k.npy'


import numpy as np
from skimage.feature import hog

# Load the numpy file containing the images
images = np.load('C:/Users/Hamza/Desktop/saved_dataset/image5k.npy')
#images = np.load('C:/Users/Hamza/Desktop/experiments/expl.npy')
# Initialize an empty list to store the HOG features
hog_features = []

# Iterate through the images
for image in images:
    # Apply HOG to the current image and append the resulting feature vector to the list
    hog_features.append(hog(image, channel_axis=2))
    #fd, hog_image = hog(image, visualise = True)
    
X=np.array(hog_features)
y=np.load('C:/Users/Hamza/Desktop/saved_dataset/label5k.npy')
#y=np.load('C:/Users/Hamza/Desktop/experiments/expi.npy')

start=time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)




model_name="lda"

def evaluate_model(model_name,X,y):
    if(model_name=="lda"):
        lda=LinearDiscriminantAnalysis()

        lda.fit(X_train,y_train)

        accuracy=lda.score(X_test,y_test)
        print("Accuracy of LDA: {:.2f}%".format(accuracy * 100))

        with open('lda.pkl', 'wb') as file:
            model=pickle.dump(lda, file)

    elif(model_name=="nearest_centroid"):
        nc = NearestCentroid()

        # Train the classifier on the training data
        nc.fit(X_train, y_train)

        # Evaluate the classifier on the test data
        accuracy = nc.score(X_test, y_test)
        print("Accuracy of nearest centroid: {:.2f}%".format(accuracy * 100))

        with open('nearest_centroid.pkl', 'wb') as file:
            model=pickle.dump(nc, file)
        
    elif(model_name=="RandomForestClassifier"):
        rf=RandomForestClassifier()

        rf.fit(X_train,y_train)

        accuracy=rf.score(X_test,y_test)
        print("Accuracy of random forest: {:.2f}%".format(accuracy * 100))

        with open('random_forest.pkl', 'wb') as file:
            model=pickle.dump(rf, file)
        
    elif(model_name=="LogisticRegression"):
        lr = LogisticRegression() 

        # Train the model on the training data
        lr.fit(X_train, y_train)

        # Evaluate the model on the test data
        accuracy = lr.score(X_test, y_test)
        print("Accuracy of logistic regression : {:.2f}%".format(accuracy * 100))

        with open('logistic_regression.pkl', 'wb') as file:
            model=pickle.dump(lr, file)

    elif(model_name=="SVC"):
        with open('C:/Users/Hamza/Desktop/project/Machine_learning/saved_models/svc.pkl', 'rb') as file:
            model = pickle.load(file)
            
    elif(model_name=="KNN"):
        with open('C:/Users/Hamza/Desktop/project/Machine_learning/saved_models/knn.pkl', 'rb') as file:
            model = pickle.load(file)

    elif(model_name=="DecisionTree"):
        with open('C:/Users/Hamza/Desktop/project/Machine_learning/saved_models/decisionTree.pkl', 'rb') as file:
            model = pickle.load(file)
    
    elif(model_name=="bernoulliNB"):
        with open('C:/Users/Hamza/Desktop/project/Machine_learning/saved_models/bernoulli.pkl', 'rb') as file:
            model = pickle.load(file)
    
    elif(model_name=="gaussianNB"):
        with open('C:/Users/Hamza/Desktop/project/Machine_learning/saved_models/GuassianNB.pkl', 'rb') as file:
            model = pickle.load(file)
        
    elif(model_name=="passive_aggressive"):
        with open('C:/Users/Hamza/Desktop/project/Machine_learning/saved_models/passive_ac.pkl', 'rb') as file:
            model = pickle.load(file)

    elif(model_name=="perceptron"):
        with open('C:/Users/Hamza/Desktop/project/Machine_learning/saved_models/perceptron.pkl.pkl', 'rb') as file:
            model = pickle.load(file)
            
    elif(model_name=="SGDC"):
        with open('C:/Users/Hamza/Desktop/project/Machine_learning/saved_models/sgdc.pkl', 'rb') as file:
            model = pickle.load(file)

    elif(model_name=="ridge_classifier"):
        with open('C:/Users/Hamza/Desktop/project/Machine_learning/saved_models/ridge.pkl', 'rb') as file:
            model = pickle.load(file)
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    print("Accuracy of ",model,":"" {:.2f}%".format(accuracy * 100))
    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.imshow(conf_matrix, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            text = conf_matrix[i][j]
            plt.text(j, i, text, ha="center",fontsize=15, va="center", color="black")

    # Save the figure
    fig_lda = plt.gcf()
    confusion_matrix_plot = fig_lda
    confusion_matrix_plot.savefig("confusion_matrix_LDA.png")
    plt.show()  
    
evaluate_model(model_name, X, y)