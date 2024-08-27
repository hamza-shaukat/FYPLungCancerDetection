# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 21:14:38 2023

@author: Hamza
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 17:28:10 2023

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
#images = np.load('C:/Users/Hamza/Desktop/saved_dataset/image5k.npy')
images = np.load('C:/Users/Hamza/Desktop/experiments/expl.npy')
# Initialize an empty list to store the HOG features
hog_features = []

# Iterate through the images
for image in images:
    # Apply HOG to the current image and append the resulting feature vector to the list
    hog_features.append(hog(image, channel_axis=2))
    #fd, hog_image = hog(image, visualise = True)
    
X=np.array(hog_features)
#y=np.load('C:/Users/Hamza/Desktop/saved_dataset/label5k.npy')
y=np.load('C:/Users/Hamza/Desktop/experiments/expi.npy')

start=time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)




with open('lda.pkl', 'rb') as file:
    model = pickle.load(file)
lda=LinearDiscriminantAnalysis()    


predictions = model.predict(X_test)

y_pred = lda.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
#%%




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



























