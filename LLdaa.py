# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 17:52:43 2023

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
import datetime
#-------
start=time.time()
folder_path = 'C:/Users/Hamza/Desktop/dataset5k'

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
end=time.time()
print("time taken in feature extraction= ",end-start,"seconds")

start=time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)


model=LinearDiscriminantAnalysis()

model.fit(X_train,y_train)

accuracy=model.score(X_test,y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))
end=time.time()
print("time taken in classification= ",end-start,"seconds")
