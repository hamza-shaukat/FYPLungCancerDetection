# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 00:42:48 2023

@author: Hamza
"""

import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

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

root.mainloop()