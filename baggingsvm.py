# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 19:29:39 2023

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
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

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
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)


base_estimator=svm.SVC(kernel='linear', probability=True)
bagging = BaggingClassifier(base_estimator=base_estimator, n_estimators=10)
bagging.fit(X_train,y_train)
predictions = bagging.predict(X_test)
accuracy=bagging.score(X_test,y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))


