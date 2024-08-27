# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 18:39:51 2023

@author: Hamza
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 14:56:09 2023

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
from sklearn.metrics import classification_report

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




from skimage.feature import hog


import tkinter as tk
from tkinter import filedialog, messagebox, Label, ttk
from PIL import ImageTk, Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Create the main window
root = tk.Tk()
root.title("Lung Cancer Image Classifier")
root.geometry("800x600")
root.configure(bg='#F0F0F0') # background color

# Create a label for the title
title = Label(root, text="Lung Cancer Detection", font=("Arial", 20), bg='#FFFFFF')
title.pack(pady=10)

# Create IntVar for the radio buttons
ml_var = tk.IntVar()
dl_var = tk.IntVar()

# Create radio button for machine learning models
ml_radio = tk.Radiobutton(root, text="Machine Learning Models", variable=ml_var, value=1, command=lambda: set_models("ML"), bg='white')
ml_radio.pack(pady=10)

# Create radio button for deep learning models
dl_radio = tk.Radiobutton(root, text="Deep Learning Models", variable=dl_var, value=2, command=lambda: set_models("DL"), bg='white')
dl_radio.pack(pady=10)

# Create a tkinter StringVar to store the selected model
model_var = tk.StringVar()

# Create a combobox to display the models
model_combobox = ttk.Combobox(root, textvariable=model_var)
model_combobox.pack(pady=10)

# Function to set the models in the combobox based on the selected radio button
def set_models(model_type):
    models_path = "C:/Users/Hamza/Desktop/exp2/modelss/"
    if model_type == "ML":
        models = ["lda", "nearest_centroid", "Random_forest","logistic_regression", "svc", "knn","decisionTree", "bernoulli", "GuassianNB","passive_ac", "perceptron", "sgdc", "ridge"]
        ml_var.set(1)
        dl_var.set(0)
    else:
        models = ["VGG16", "InceptionV3", "ResNet50", "MobileNet", "CNN", "Xception"]
        ml_var.set(0)
        dl_var.set(2)

    model_var.set(models[0])
    model_combobox['values'] = models
    
    selected_model_file = os.path.join(models_path, f"{model_var.get()}.pkl")

# Function to browse and select an image file

def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        return file_path
    else:
        messagebox.showerror("Error", "No file selected")
        return

# Function to classify the selected image
def classify_image():
    file_path = browse_file()
    if file_path:
        img = cv2.imread(file_path)

        # Resize the image
        img = cv2.resize(img, (256, 256))

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #%%
        # Reshape the image to 2D array
        gray=np.expand_dims(gray, axis=2)

        hog_features = []
        #%%

        hog_features = hog(gray, channel_axis=2)

        #%%

        X = np.array(hog_features)
        #%%

        with open('C:/Users/Hamza/Desktop/exp2/models/'+ model_var.get()+'.pkl', 'rb') as file:
            model = pickle.load(file)

        #%%
        prediction = model.predict(X.reshape(1, -1))# Insert code here to pass the image through the selected model
        # and get the prediction
        
        model = model_var.get()
        if(prediction=='0'):
            prediction='negative'
        elif(prediction=='1'):
            prediction='positive'

        result_label.config(text=prediction)
        img = Image.open(file_path)
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img


    

# Create browse button
browse_button = tk.Button(root, text="Browse", command=classify_image, bg='white')
browse_button.pack(pady=10)

# Create a main frame to hold all the widgets
main_frame = tk.Frame(root)
main_frame.pack(pady=20)

# Create frame for selected image and result
image_frame = tk.Frame(main_frame)
image_frame.pack(side="left", padx=20)

# Create image label
image_label = tk.Label(image_frame, text="Selected Image")
image_label.pack(side="left")

# Create result label
result_label = tk.Label(image_frame, text="", font=("Arial", 16))
result_label.pack(side="left", padx=50)



root.mainloop()