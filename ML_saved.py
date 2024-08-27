# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 21:19:52 2023

@author: Hamza
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 17:28:10 2023

@author: Hamza"""
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


#start=time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)



rf=RandomForestClassifier()

rf.fit(X_train,y_train)

accuracy=rf.score(X_test,y_test)
print("Accuracy of random forest: {:.2f}%".format(accuracy * 100))

with open('rf.pkl', 'wb') as file:
    model=pickle.dump(rf, file)





lda=LinearDiscriminantAnalysis()

lda.fit(X_train,y_train)

accuracy=lda.score(X_test,y_test)
print("Accuracy of LDA: {:.2f}%".format(accuracy * 100))

with open('lda.pkl', 'wb') as file:
    model=pickle.dump(lda, file)
    
    



sv=svm.SVC(kernel='sigmoid')

sv.fit(X_train,y_train)

accuracy=sv.score(X_test,y_test)
print("Accuracy of svc: {:.2f}%".format(accuracy * 100))


with open('svc.pkl', 'wb') as file:
    model=pickle.dump(sv, file)




dt=DecisionTreeClassifier()

dt.fit(X_train,y_train)

accuracy=dt.score(X_test,y_test)
print("Accuracy of decision tree: {:.2f}%".format(accuracy * 100))

with open('decisionTree.pkl', 'wb') as file:
    model=pickle.dump(dt, file)




knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = knn.score(X_test, y_test)
print("Accuracy of KNN: {:.2f}%".format(accuracy * 100))

with open('knn.pkl', 'wb') as file:
    model=pickle.dump(knn, file)




gnb = GaussianNB()

# Train the classifier on the training data
gnb.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = gnb.score(X_test, y_test)
print("Accuracy of guassian NB: {:.2f}%".format(accuracy * 100))

with open('GuassianNB.pkl', 'wb') as file:
    model=pickle.dump(gnb, file)





prcp = Perceptron()

# Train the classifier on the training data
prcp.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = prcp.score(X_test, y_test)
print("Accuracy of perceptron: {:.2f}%".format(accuracy * 100))

with open('perceptron.pkl', 'wb') as file:
    model=pickle.dump(prcp, file)





lr = LogisticRegression() 

# Train the model on the training data
lr.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = lr.score(X_test, y_test)
print("Accuracy of logistic regression : {:.2f}%".format(accuracy * 100))

with open('logistic_regression.pkl', 'wb') as file:
    model=pickle.dump(lr, file)





sgdc = SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=42)

# Train the model on the training data
sgdc.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = sgdc.score(X_test, y_test)
print("Accuracy of SGDC: {:.2f}%".format(accuracy * 100))


with open('sgdc.pkl', 'wb') as file:
    model=pickle.dump(sgdc, file)




nc = NearestCentroid()

# Train the classifier on the training data
nc.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = nc.score(X_test, y_test)
print("Accuracy of nearest centroid: {:.2f}%".format(accuracy * 100))

with open('nearest_centroid.pkl', 'wb') as file:
    model=pickle.dump(nc, file)



pac = PassiveAggressiveClassifier()

# Train the classifier on the training data
pac.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = pac.score(X_test, y_test)
print("Accuracy of passive aggressive classifier: {:.2f}%".format(accuracy * 100))

with open('passive_ac.pkl', 'wb') as file:
    model=pickle.dump(pac, file)




bnb = BernoulliNB()

# Train the classifier on the training data
bnb.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = bnb.score(X_test, y_test)
print("Accuracy of bernoulliNB: {:.2f}%".format(accuracy * 100))

with open('bernoulli.pkl', 'wb') as file:
    model=pickle.dump(bnb, file)
    


# Create a Ridge Classifier
rc = RidgeClassifier()

# Train the classifier on the training data
rc.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = rc.score(X_test, y_test)
print("Accuracy of ridge classifier: {:.2f}%".format(accuracy * 100))
with open('ridge.pkl', 'wb') as file:
    model=pickle.dump(rc, file)
