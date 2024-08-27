# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 22:08:39 2023

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
import pickle
from skimage.feature import hog
#-------

folder_path = 'C:/Users/Hamza/Desktop/dataset5kk'

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
    label_features.append(ir_)
    print(i+1, '/' , total_images,'-->',round((i+1)/total_images*100,4),'%')
X=np.array(image_features)
y=np.array(label_features)

#save in file
np.save('C:/Users/Hamza/Desktop/saved data/x5k.npy', X)
np.save('C:/Users/Hamza/Desktop/saved data/y5k.npy', y)

#load from file
#X = np.load('C:/Users/Hamza/Desktop/saved data/xt.npy')
#y = np.load('C:/Users/Hamza/Desktop/saved data/yt.npy')


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)


'''rf=RandomForestClassifier()

rf.fit(X_train,y_train)

accuracy=rf.score(X_test,y_test)
print("Accuracy of random forest: {:.2f}%".format(accuracy * 100))


lda=LinearDiscriminantAnalysis()

lda.fit(X_train,y_train)

accuracy=lda.score(X_test,y_test)
print("Accuracy of LDA: {:.2f}%".format(accuracy * 100))
#end=time.time()
#print("time taken in classification= ",end-start,"seconds")


sv=svm.SVC(kernel='sigmoid')

sv.fit(X_train,y_train)

accuracy=sv.score(X_test,y_test)
print("Accuracy of svc: {:.2f}%".format(accuracy * 100))


dt=DecisionTreeClassifier()

dt.fit(X_train,y_train)

accuracy=dt.score(X_test,y_test)
print("Accuracy of decision tree: {:.2f}%".format(accuracy * 100))


knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = knn.score(X_test, y_test)
print("Accuracy of KNN: {:.2f}%".format(accuracy * 100))



gnb = GaussianNB()

# Train the classifier on the training data
gnb.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = gnb.score(X_test, y_test)
print("Accuracy of guassian NB: {:.2f}%".format(accuracy * 100))




prcp = Perceptron()

# Train the classifier on the training data
prcp.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = prcp.score(X_test, y_test)
print("Accuracy of perceptron: {:.2f}%".format(accuracy * 100))


'''
lr = LogisticRegression() # ye tha

# Train the model on the training data
lr.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = lr.score(X_test, y_test)
print("Accuracy of logistic regression : {:.2f}%".format(accuracy * 100))



sgdc = SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=42)

# Train the model on the training data
sgdc.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = sgdc.score(X_test, y_test)
print("Accuracy of SGDC: {:.2f}%".format(accuracy * 100))


with open('sgdc.pkl', 'rb') as file:
    model=pickle.dump(file)

nc = NearestCentroid()

# Train the classifier on the training data
nc.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = nc.score(X_test, y_test)
print("Accuracy of nearest centroid: {:.2f}%".format(accuracy * 100))

with open('nc.pkl', 'rb') as file:
    model=pickle.dump(file)

pac = PassiveAggressiveClassifier()

# Train the classifier on the training data
pac.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = pac.score(X_test, y_test)
print("Accuracy of passive aggressive classifier: {:.2f}%".format(accuracy * 100))

with open('pac.pkl', 'rb') as file:
    model=pickle.dump(file)

bnb = BernoulliNB()

# Train the classifier on the training data
bnb.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = bnb.score(X_test, y_test)
print("Accuracy of bernoulliNB: {:.2f}%".format(accuracy * 100))

with open('bnb.pkl', 'rb') as file:
    model=pickle.dump(file)
#model = svm.SVC(probability=True)

# Create a CalibratedClassifierCV
#cc = CalibratedClassifierCV(svm.svc, cv=5, method='sigmoid')

# Train the classifier on the training data
#cc.fit(X_train, y_train)

# Evaluate the classifier on the test data
#accuracy = cc.score(X_test, y_test)
#print("Accuracy of callibrated classifier: {:.2f}%".format(accuracy * 100))



# Create a LightGBM dataset
#lgb_train = lgb.Dataset(X_train, label=y_train)

# Define the LightGBM model parameters
#params = {
#   'boosting_type': 'gbdt',
#    'objective': 'binary',
#    'metric': 'binary_logloss',
#    'num_leaves': 31,
#    'learning_rate': 0.05,
#    'feature_fraction': 0.9,
#    'bagging_fraction': 0.8,
#    'bagging_freq': 5,
#    'verbose': 0
#}

# Train the LightGBM model
#gbm = lgb.train(params,
 #               lgb_train,
 #               num_boost_round=100)

# Make predictions on the test data
#y_pred = gbm.predict(X_test)

# Convert the predicted probabilities to binary labels
#y_pred = [1 if p > 0.5 else 0 for p in y_pred]
#accuracy = gbm.score(y_test, y_pred)
#print("Accuracy of LGBM: {:.2f}%".format(accuracy * 100))


# Create a Ridge Classifier
rc = RidgeClassifier()

# Train the classifier on the training data
rc.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = rc.score(X_test, y_test)
print("Accuracy of ridge classifier: {:.2f}%".format(accuracy * 100))
with open('ric.pkl', 'rb') as file:
    model=pickle.dump(file)



