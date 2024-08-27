import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,Input,Add
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import shutil
from tkinter import filedialog
from tkinter import messagebox

from tensorflow.keras import optimizers
import glob
import numpy as np
import sys
from tensorflow.keras.callbacks import Callback, CSVLogger,ModelCheckpoint;
#from livelossplot import PlotLossesKeras
from time import time
import json
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
#np.random.seed(1000)
folder_path = 'C:/Users/Hamza/Desktop/dataset50'

folder_images = glob(folder_path + '/*/*.jpg')


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


