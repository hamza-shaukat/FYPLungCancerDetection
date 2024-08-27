# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 13:27:26 2023

@author: Hamza
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 19:56:49 2023

@author: Hamza
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 22:08:39 2023

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
#-------

#folder_path = 'C:/Users/Hamza/Desktop/dataset50'

#folder_images = glob(folder_path + '/*/*.jpg')


#hog = cv2.HOGDescriptor()
'''from skimage.feature import hog
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
np.save('C:/Users/Hamza/Desktop/saved data/xt.npy', X)
np.save('C:/Users/Hamza/Desktop/saved data/yt.npy', y)'''
X = np.load('C:/Users/Hamza/Desktop/saved data/xt.npy')
y = np.load('C:/Users/Hamza/Desktop/saved data/yt.npy')
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)


rf=RandomForestClassifier()

#rf.fit(X_train,y_train)

accuracy=rf.score(X_test,y_test)
print("Accuracy of random forest: {:.2f}%".format(accuracy * 100))

actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()


lda=LinearDiscriminantAnalysis()

#lda.fit(X_train,y_train)

accuracy=lda.score(X_test,y_test)
print("Accuracy of LDA: {:.2f}%".format(accuracy * 100))

with open('lda.pkl', 'wb') as file:
    model=pickle.dump(lda, file)
    
    
actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()
#end=time.time()
#print("time taken in classification= ",end-start,"seconds")


sv=svm.SVC(kernel='sigmoid')

#sv.fit(X_train,y_train)

accuracy=sv.score(X_test,y_test)
print("Accuracy of svc: {:.2f}%".format(accuracy * 100))


with open('svc.pkl', 'wb') as file:
    model=pickle.dump(sv, file)

actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()



dt=DecisionTreeClassifier()

#dt.fit(X_train,y_train)

accuracy=dt.score(X_test,y_test)
print("Accuracy of decision tree: {:.2f}%".format(accuracy * 100))

with open('decisionTree.pkl', 'wb') as file:
    model=pickle.dump(dt, file)


actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()



knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifier on the training data
#knn.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = knn.score(X_test, y_test)
print("Accuracy of KNN: {:.2f}%".format(accuracy * 100))

with open('knn.pkl', 'wb') as file:
    model=pickle.dump(knn, file)

actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()



gnb = GaussianNB()

# Train the classifier on the training data
#gnb.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = gnb.score(X_test, y_test)
print("Accuracy of guassian NB: {:.2f}%".format(accuracy * 100))

with open('GuassianNB.pkl', 'wb') as file:
    model=pickle.dump(gnb, file)

actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()




prcp = Perceptron()

# Train the classifier on the training data
#prcp.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = prcp.score(X_test, y_test)
print("Accuracy of perceptron: {:.2f}%".format(accuracy * 100))

with open('perceptron.pkl', 'wb') as file:
    model=pickle.dump(prcp, file)

actual_svm = numpy.random.binomial(1,.9,size = 1000)
predicted_svm = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()





lr = LogisticRegression() 

# Train the model on the training data
#lr.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = lr.score(X_test, y_test)
print("Accuracy of logistic regression : {:.2f}%".format(accuracy * 100))

with open('logistic_regression.pkl', 'wb') as file:
    model=pickle.dump(lr, file)

actual_svm = numpy.random.binomial(1,.9,size = 1000)
predicted_svm = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()




sgdc = SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=42)

# Train the model on the training data
#sgdc.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = sgdc.score(X_test, y_test)
print("Accuracy of SGDC: {:.2f}%".format(accuracy * 100))


with open('sgdc.pkl', 'wb') as file:
    model=pickle.dump(sgdc, file)

actual_svm = numpy.random.binomial(1,.9,size = 1000)
predicted_svm = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()



nc = NearestCentroid()

# Train the classifier on the training data
#nc.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = nc.score(X_test, y_test)
print("Accuracy of nearest centroid: {:.2f}%".format(accuracy * 100))

with open('nearest_centroid.pkl', 'wb') as file:
    model=pickle.dump(nc, file)

actual_svm = numpy.random.binomial(1,.9,size = 1000)
predicted_svm = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()


pac = PassiveAggressiveClassifier()

# Train the classifier on the training data
#pac.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = pac.score(X_test, y_test)
print("Accuracy of passive aggressive classifier: {:.2f}%".format(accuracy * 100))

with open('passive_ac.pkl', 'wb') as file:
    model=pickle.dump(pac, file)


   


actual_svm = numpy.random.binomial(1,.9,size = 1000)
predicted_svm = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()


bnb = BernoulliNB()

# Train the classifier on the training data
#bnb.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = bnb.score(X_test, y_test)
print("Accuracy of bernoulliNB: {:.2f}%".format(accuracy * 100))

with open('bernoulli.pkl', 'wb') as file:
    model=pickle.dump(bnb, file)
    
actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.title="Bernouli"
plt.show()



# Create a Ridge Classifier
rc = RidgeClassifier()

# Train the classifier on the training data
#rc.fit(X_train, y_train)

# Evaluate the classifier on the test data
accuracy = rc.score(X_test, y_test)
print("Accuracy of ridge classifier: {:.2f}%".format(accuracy * 100))
with open('ridge.pkl', 'wb') as file:
    model=pickle.dump(rc, file)
    
actual_svm = numpy.random.binomial(1,.9,size = 1000)
predicted_svm = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()


def evaluate_model(model_name,x,y):
    if(model_name=="PassiveAggressiveClassifier"):
        mod=with open('pac.pkl', 'rb') as file:
        model = pickle.load('passive_ac.pkl')

    else if(model_name=="nearest_centroid"):
        mod=mod=with open('passive_ac.pkl', 'rb') as file:
        model = pickle.load('passive_ac.pkl')
        
        
    else if(model_name=="RandomForestClassifier"):
         
        mod=with open('passive_ac.pkl.pkl', 'rb') as file:
        model = pickle.load('passive_ac.pkl')
        
    else if(model_name=="nearest_centroid"):
         mod=mod=with open('passive_ac.pkl', 'rb') as file:
         model = pickle.load('passive_ac.pkl')
         
    else if(model_name=="nearest_centroid"):
          mod=mod=with open('passive_ac.pkl', 'rb') as file:
          model = pickle.load('passive_ac.pkl')

    else if(model_name=="nearest_centroid"):
          mod=mod=with open('passive_ac.pkl', 'rb') as file:
          model = pickle.load('passive_ac.pkl')  
          
    else if(model_name=="nearest_centroid"):
         mod=mod=with open('passive_ac.pkl', 'rb') as file:
         model = pickle.load('passive_ac.pkl')
    
    else if(model_name=="nearest_centroid"):
        mod=mod=with open('passive_ac.pkl', 'rb') as file:
        model = pickle.load('passive_ac.pkl')
    
    else if(model_name=="nearest_centroid"):
        mod=mod=with open('passive_ac.pkl', 'rb') as file:
        model = pickle.load('passive_ac.pkl')
        
    else if(model_name=="nearest_centroid"):
        mod=mod=with open('passive_ac.pkl', 'rb') as file:
        model = pickle.load('passive_ac.pkl')
    
    else if(model_name=="nearest_centroid"):
        mod=mod=with open('passive_ac.pkl', 'rb') as file:
        model = pickle.load('passive_ac.pkl')    
    y_pred=mod.predict(x)
 
