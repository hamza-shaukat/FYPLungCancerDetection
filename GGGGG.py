# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 14:56:09 2023

@author: Hamza
"""

import tkinter as tk
from tkinter import filedialog, messagebox, Label, ttk
from PIL import ImageTk, Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Create the main window
root = tk.Tk()
root.title("Lung Cancer Image Classifier")
root.geometry("800x600")
root.configure(bg='#FFFFFF') # background color

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
    if model_type == "ML":
        models = ["lda", "nearest_centroid", "RandomForestClassifier","LogisticRegression", "SVC", "KNN","DecisionTree", "bernoulliNB", "gaussianNB","passive_aggressive", "perceptron", "SGDC", "ridge_classifier"]
        ml_var.set(1)
        dl_var.set(0)
    else:
        models = ["VGG16", "InceptionV3", "ResNet50", "MobileNet", "DenseNet", "Xception"]
        ml_var.set(0)
        dl_var.set(2)

    model_var.set(models[0])
    model_combobox['values'] = models

# Function to browse and select an image file
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
        # Insert code here to pass the image through the selected model
        # and get the prediction
        model = model_var.get()
        if model == "Model 1":
            prediction = "Positive"  # Example prediction for Model 1
        elif model == "Model 2":
            prediction = "Negative"  # Example prediction for Model 2
        else:
            prediction = "Unknown"  # Example prediction for Model 3
        result_label.config(text=prediction)
        img = Image.open(file_path)
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        # Insert code here to generate the confusion matrix
        y_true = ["Negative", "Positive", "Negative", "Positive"]
        y_pred = ["Negative", "Positive", "Positive", "Negative"]
        cm = confusion_matrix(y_true, y_pred)
        classification = classification_report(y_true, y_pred)
        # Insert code here to display the confusion matrix and metrics
        metrics_label.config(text=classification)
        cm_image = Image.fromarray(np.uint8(cm))
        cm_image = cm_image.resize((250,250), Image.ANTIALIAS)
        cm_image = ImageTk.PhotoImage(cm_image)
        confusion_label.config(image=cm_image)
        confusion_label.image = cm_image




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
image_label = tk.Label(image_frame)
image_label.pack()

# Create result label
result_label = tk.Label(image_frame, text="", font=("Arial", 16))
result_label.pack(pady=10)

# Create frame for metrics and confusion matrix
metrics_frame = tk.Frame(main_frame)
metrics_frame.pack(side="right", padx=20)

# Create metrics label
metrics_label = tk.Label(metrics_frame, text="", font=("Arial", 10))
metrics_label.pack()
confusion_label = tk.Label(metrics_frame)
confusion_label.pack()

root.mainloop()