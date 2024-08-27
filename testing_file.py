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





lda=LinearDiscriminantAnalysis()

lda.fit(X_train,y_train)

accuracy=lda.score(X_test,y_test)
print("Accuracy of LDA: {:.2f}%".format(accuracy * 100))

with open('lda.pkl', 'wb') as file:
    model=pickle.dump(lda, file)
    

y_pred = lda.predict(X_test)
#y_pred= np.expand_dims(y_pred, axis=-1)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
#This code will predict the labels for the test data using the trained Random Forest model and then calculate the confusion matrix by comparing the predicted labels with the true labels stored in y_test. You can also use other visualization libraries like matplotlib or seaborn to plot the confusion matrix in a more graphical format.


import matplotlib.pyplot as plt

plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save the figure
fig_lda = plt.gcf()
confusion_matrix_plot = fig_lda
plt.show()


rf=RandomForestClassifier()

rf.fit(X_train,y_train)

accuracy=rf.score(X_test,y_test)
print("Accuracy of random forest: {:.2f}%".format(accuracy * 100))

with open('random_forest.pkl', 'wb') as file:
    model=pickle.dump(rf, file)



y_pred = rf.predict(X_test)
#y_pred= np.expand_dims(y_pred, axis=-1)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
#This code will predict the labels for the test data using the trained Random Forest model and then calculate the confusion matrix by comparing the predicted labels with the true labels stored in y_test. You can also use other visualization libraries like matplotlib or seaborn to plot the confusion matrix in a more graphical format.


import matplotlib.pyplot as plt

plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save the figure
fig_rf = plt.gcf()
confusion_matrix_plot = fig_rf
plt.show()



sv=svm.SVC(kernel='sigmoid')

sv.fit(X_train,y_train)

accuracy=sv.score(X_test,y_test)
print("Accuracy of svc: {:.2f}%".format(accuracy * 100))


with open('svc.pkl', 'wb') as file:
    model=pickle.dump(sv, file)
    
  
y_pred = sv.predict(X_test)
#y_pred= np.expand_dims(y_pred, axis=-1)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
#This code will predict the labels for the test data using the trained Random Forest model and then calculate the confusion matrix by comparing the predicted labels with the true labels stored in y_test. You can also use other visualization libraries like matplotlib or seaborn to plot the confusion matrix in a more graphical format.


import matplotlib.pyplot as plt

plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save the figure
fig_svm = plt.gcf()
confusion_matrix_plot = fig_svm
plt.show()    



dt=DecisionTreeClassifier()

dt.fit(X_train,y_train)

accuracy=dt.score(X_test,y_test)
print("Accuracy of decision tree: {:.2f}%".format(accuracy * 100))

with open('decisionTree.pkl', 'wb') as file:
    model=pickle.dump(dt, file)
    

y_pred = dt.predict(X_test)
#y_pred= np.expand_dims(y_pred, axis=-1)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
#This code will predict the labels for the test data using the trained Random Forest model and then calculate the confusion matrix by comparing the predicted labels with the true labels stored in y_test. You can also use other visualization libraries like matplotlib or seaborn to plot the confusion matrix in a more graphical format.


import matplotlib.pyplot as plt

plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save the figure
fig_dt = plt.gcf()
confusion_matrix_plot = fig_dt
plt.show()


    
knn = KNeighborsClassifier(n_neighbors=5)

 # Train the classifier on the training data
knn.fit(X_train, y_train)

 # Evaluate the classifier on the test data
accuracy = knn.score(X_test, y_test)
print("Accuracy of KNN: {:.2f}%".format(accuracy * 100))

with open('knn.pkl', 'wb') as file:
     model=pickle.dump(knn, file)   


y_pred = knn.predict(X_test)
#y_pred= np.expand_dims(y_pred, axis=-1)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
#This code will predict the labels for the test data using the trained Random Forest model and then calculate the confusion matrix by comparing the predicted labels with the true labels stored in y_test. You can also use other visualization libraries like matplotlib or seaborn to plot the confusion matrix in a more graphical format.


import matplotlib.pyplot as plt

plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save the figure
fig_knn = plt.gcf()
confusion_matrix_plot = fig_knn
plt.show()



gnb = GaussianNB()

# Train the classifier on the training data
gnb.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = gnb.score(X_test, y_test)
print("Accuracy of guassian NB: {:.2f}%".format(accuracy * 100))

with open('GuassianNB.pkl', 'wb') as file:
    model=pickle.dump(gnb, file)


y_pred = gnb.predict(X_test)
#y_pred= np.expand_dims(y_pred, axis=-1)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
#This code will predict the labels for the test data using the trained Random Forest model and then calculate the confusion matrix by comparing the predicted labels with the true labels stored in y_test. You can also use other visualization libraries like matplotlib or seaborn to plot the confusion matrix in a more graphical format.


import matplotlib.pyplot as plt

plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save the figure
fig_gnb = plt.gcf()
confusion_matrix_plot = fig_gnb
plt.show()



prcp = Perceptron()

# Train the classifier on the training data
prcp.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = prcp.score(X_test, y_test)
print("Accuracy of perceptron: {:.2f}%".format(accuracy * 100))

with open('perceptron.pkl', 'wb') as file:
    model=pickle.dump(prcp, file)



y_pred = prcp.predict(X_test)
#y_pred= np.expand_dims(y_pred, axis=-1)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
#This code will predict the labels for the test data using the trained Random Forest model and then calculate the confusion matrix by comparing the predicted labels with the true labels stored in y_test. You can also use other visualization libraries like matplotlib or seaborn to plot the confusion matrix in a more graphical format.


import matplotlib.pyplot as plt

plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save the figure
fig_prcp = plt.gcf()
confusion_matrix_plot = fig_prcp
plt.show()



lr = LogisticRegression() 

# Train the model on the training data
lr.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = lr.score(X_test, y_test)
print("Accuracy of logistic regression : {:.2f}%".format(accuracy * 100))

with open('logistic_regression.pkl', 'wb') as file:
    model=pickle.dump(lr, file)


y_pred = lr.predict(X_test)
#y_pred= np.expand_dims(y_pred, axis=-1)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
#This code will predict the labels for the test data using the trained Random Forest model and then calculate the confusion matrix by comparing the predicted labels with the true labels stored in y_test. You can also use other visualization libraries like matplotlib or seaborn to plot the confusion matrix in a more graphical format.


import matplotlib.pyplot as plt

plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save the figure
fig_lr = plt.gcf()
confusion_matrix_plot = fig_lr
plt.show()





sgdc = SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=42)

# Train the model on the training data
sgdc.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = sgdc.score(X_test, y_test)
print("Accuracy of SGDC: {:.2f}%".format(accuracy * 100))


with open('sgdc.pkl', 'wb') as file:
    model=pickle.dump(sgdc, file)


y_pred = sgdc.predict(X_test)
#y_pred= np.expand_dims(y_pred, axis=-1)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
#This code will predict the labels for the test data using the trained Random Forest model and then calculate the confusion matrix by comparing the predicted labels with the true labels stored in y_test. You can also use other visualization libraries like matplotlib or seaborn to plot the confusion matrix in a more graphical format.


import matplotlib.pyplot as plt

plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save the figure
fig_sgdc = plt.gcf()
confusion_matrix_plot = fig_sgdc
plt.show()



nc = NearestCentroid()

# Train the classifier on the training data
nc.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = nc.score(X_test, y_test)
print("Accuracy of nearest centroid: {:.2f}%".format(accuracy * 100))

with open('nearest_centroid.pkl', 'wb') as file:
    model=pickle.dump(nc, file)

y_pred = nc.predict(X_test)
#y_pred= np.expand_dims(y_pred, axis=-1)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
#This code will predict the labels for the test data using the trained Random Forest model and then calculate the confusion matrix by comparing the predicted labels with the true labels stored in y_test. You can also use other visualization libraries like matplotlib or seaborn to plot the confusion matrix in a more graphical format.


import matplotlib.pyplot as plt

plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save the figure
fig_nc = plt.gcf()
confusion_matrix_plot = fig_nc
plt.show()



pac = PassiveAggressiveClassifier()

# Train the classifier on the training data
pac.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = pac.score(X_test, y_test)
print("Accuracy of passive aggressive classifier: {:.2f}%".format(accuracy * 100))

with open('passive_ac.pkl', 'wb') as file:
    model=pickle.dump(pac, file)


y_pred = pac.predict(X_test)
#y_pred= np.expand_dims(y_pred, axis=-1)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
#This code will predict the labels for the test data using the trained Random Forest model and then calculate the confusion matrix by comparing the predicted labels with the true labels stored in y_test. You can also use other visualization libraries like matplotlib or seaborn to plot the confusion matrix in a more graphical format.


import matplotlib.pyplot as plt

plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save the figure
fig_pac = plt.gcf()
confusion_matrix_plot = fig_pac
plt.show()




bnb = BernoulliNB()

# Train the classifier on the training data
bnb.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = bnb.score(X_test, y_test)
print("Accuracy of bernoulliNB: {:.2f}%".format(accuracy * 100))

with open('bernoulli.pkl', 'wb') as file:
    model=pickle.dump(bnb, file)


y_pred = bnb.predict(X_test)
#y_pred= np.expand_dims(y_pred, axis=-1)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
#This code will predict the labels for the test data using the trained Random Forest model and then calculate the confusion matrix by comparing the predicted labels with the true labels stored in y_test. You can also use other visualization libraries like matplotlib or seaborn to plot the confusion matrix in a more graphical format.


import matplotlib.pyplot as plt

plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save the figure
fig_bnb = plt.gcf()
confusion_matrix_plot = fig_bnb
plt.show()


rc = RidgeClassifier()

# Train the classifier on the training data
rc.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = rc.score(X_test, y_test)
print("Accuracy of ridge classifier: {:.2f}%".format(accuracy * 100))
with open('ridge.pkl', 'wb') as file:
    model=pickle.dump(rc, file)


y_pred = rc.predict(X_test)
#y_pred= np.expand_dims(y_pred, axis=-1)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
#This code will predict the labels for the test data using the trained Random Forest model and then calculate the confusion matrix by comparing the predicted labels with the true labels stored in y_test. You can also use other visualization libraries like matplotlib or seaborn to plot the confusion matrix in a more graphical format.


import matplotlib.pyplot as plt

plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save the figure
fig_rc = plt.gcf()
confusion_matrix_plot = fig_rc
plt.show()












































#%%

from sklearn.metrics import confusion_matrix

y_pred = rf.predict(X_test)
y_pred= np.expand_dims(y_pred, axis=-1)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
#This code will predict the labels for the test data using the trained Random Forest model and then calculate the confusion matrix by comparing the predicted labels with the true labels stored in y_test. You can also use other visualization libraries like matplotlib or seaborn to plot the confusion matrix in a more graphical format.


import matplotlib.pyplot as plt

plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save the figure
fig = plt.gcf()
confusion_matrix_plot = fig
plt.show()


'''

from sklearn.metrics import classification_report

y_pred = rf.predict(X_test)
y_pred= np.expand_dims(y_pred, axis=-1)
print(classification_report(y_test, y_pred))


accuracy = (conf_matrix[0][0] + conf_matrix[1][1]) / conf_matrix.sum()
print("Accuracy: {:.2f}%".format(accuracy * 100))


precision = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])
recall = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[1][0])
f1_score = 2 * (precision * recall) / (precision + recall)

print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1-Score: {:.2f}%".format(f1_score * 100))
'''