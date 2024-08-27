# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 22:13:47 2023

@author: Hamza
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import PhotoImage

def load_file1():
    filepath = filedialog.askopenfilename()
    label1.config(text=filepath)

def load_file2():
    filepath = filedialog.askopenfilename()
    label2.config(text=filepath)

def test_function():
    print("Test button pressed")

root = tk.Tk()
root.title("Lung Cancer Detection")

label1 = tk.Label(root, text="File 1:")
label1.grid(row=0, column=0, padx=5, pady=5)

load_file_button1 = tk.Button(root, text="Load Model", command=load_file1)
load_file_button1.grid(row=0, column=1, padx=5, pady=5)

label2 = tk.Label(root, text="File 2:")
label2.grid(row=1, column=0, padx=5, pady=5)

load_file_button2 = tk.Button(root, text="Load Image", command=load_file2)
load_file_button2.grid(row=1, column=1, padx=5, pady=5)

label3 = tk.Label(root, text="Accuracy : ")
label3.grid(row=2, column=0, padx=5, pady=5)

text_field1 = tk.Entry(root)
text_field1.grid(row=2, column=1, padx=5, pady=5)

label4 = tk.Label(root, text="F1 Score :")
label4.grid(row=3, column=0, padx=5, pady=5)

text_field2 = tk.Entry(root)
text_field2.grid(row=3, column=1, padx=5, pady=5)

label5 = tk.Label(root, text="Text field 3:")
label5.grid(row=4, column=0, padx=5, pady=5)

text_field3 = tk.Entry(root)
text_field3.grid(row=4, column=1, padx=5, pady=5)

label6 = tk.Label(root, text="Text field 4:")
label6.grid(row=5, column=0, padx=5, pady=5)

text_field4 = tk.Entry(root)
text_field4.grid(row=5, column=1, padx=5, pady=5)

label7 = tk.Label(root, text="Text field 5:")
label7.grid(row=6, column=0, padx=5, pady=5)

text_field5 = tk.Entry(root)
text_field5.grid(row=6, column=1, padx=5, pady=5)


test_button = tk.Button(root, text="Test", command=test_function)
test_button.grid(row=7, column=0, padx=5, pady=5)


image1 = PhotoImage(file="baloon.png")
image2 = PhotoImage(file="baloon.png")
image3 = PhotoImage(file="baloon.png")

label8 = tk.Label(root, image=image1)
label8.grid(row=7, column=0, padx=5, pady=5)

label9 = tk.Label(root, image=image2)
label9.grid(row=7, column=1, padx=5, pady=5)

label10 = tk.Label(root, image=image3)
label10.grid(row=7, column=2, padx=5, pady=5)

root.mainloop()