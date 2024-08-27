# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 18:22:04 2023

@author: Hamza
"""

import tkinter as tk
from tkinter import filedialog
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

def load_data():
    filepath = filedialog.askopenfilename()
    df = pd.read_csv(filepath)
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def run_gui():
    root = tk.Tk()
    root.title("ML Project")

    load_data_button = tk.Button(root, text="Load Data", command=lambda: load_data())
    load_data_button.pack()

    train_model_button = tk.Button(root, text="Train Model", command=lambda: train_model(X_train, y_train))
    train_model_button.pack()

    evaluate_model_button = tk.Button(root, text="Evaluate Model", command=lambda: evaluate_model(model, X_test, y_test))
    evaluate_model_button.pack()

    root.mainloop()

if __name__ == '__main__':
    run_gui()