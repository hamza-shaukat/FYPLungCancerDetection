# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 00:46:18 2023

@author: Hamza
"""

import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Create the main window
root = tk.Tk()
root.title("Lung Cancer Image Classifier")

# Function to browse and select an image file
def browse_file():
    file_path = filedialog.askopenfilename()
    return file_path

# Function to classify the selected image
def classify_image():
    file_path = browse_file()
    # Insert code here to pass the image through the trained model
    # and get the prediction
    prediction = "Positive"  # Example prediction
    result_label.config(text=prediction)
    img = Image.open(file_path)
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img
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
    panel_confusion.config(image=cm_image)
    panel_confusion.image = cm_image

# Create browse button
browse_button = tk.Button(root, text="Browse", command=browse_file)
browse_button.pack()

# Create classify button
classify_button = tk.Button(root, text="Classify", command=classify_image)
classify_button.pack()

# Create result label
result_label = tk.Label(root, text="")
result_label.pack()

# Create image panel
panel = tk.Label(root)
panel.pack()

# Create confusion matrix label
metrics_label = tk.Label(root, text="")
metrics_label.pack()

# Create confusion matrix panel
panel_confusion = tk.Label(root)
panel_confusion.pack()

root.mainloop()