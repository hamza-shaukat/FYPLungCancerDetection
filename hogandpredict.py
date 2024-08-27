# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 12:32:07 2022

@author: Hamza
"""

from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def extract_features(image):
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    return fd
    # Extract HOG features using default parameters
 #   hog_features = hog(image, block)
  #  return hog_features

folder1 = 'C:/Users/Hamza/Desktop/try/0'
folder2 = 'C:/Users/Hamza/Desktop/try/1'

# Get list of file names in each folder
filenames1 = os.listdir(folder1)
filenames2 = os.listdir(folder2)

# Create full paths to each file
filepaths1 = [os.path.join(folder1, f) for f in filenames1]
filepaths2 = [os.path.join(folder2, f) for f in filenames2]


# Initialize lists to store features and labels
features = []
labels = []

# Loop through images in folder 1
for filepath in filepaths1:
    # Read image
    image = cv2.imread(filepath)
    
    # Extract features
    hog_features = extract_features(image)
    
    # Append features and label to lists
    features.append(hog_features)
    labels.append(1)
    
classifier=
##data = 
#X = data.data
#y= data.target

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)



#classifier = KNeighborsClassifier()


#X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
#classifier.fit(X_train, y_train)

# Make predictions on test data
#y_pred = classifier.predict(X_test)

# Calculate accuracy
#accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')