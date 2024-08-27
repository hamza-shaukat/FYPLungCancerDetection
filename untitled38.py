# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 19:21:52 2023

@author: Hamza
"""
import glob
import numpy as np

img = glob.glob("C:/Users/Hamza/Desktop/aaaa/*.jpg")

def median_filter(img, kernel_size):
    padded_img = np.pad(img, [(kernel_size // 2, kernel_size // 2)] * 2, mode='edge')
    filtered_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            filtered_img[i, j] = np.median(padded_img[i:i+kernel_size, j:j+kernel_size])
    return filtered_img

# Example usage
img = np.random.randint(0, 256, (100, 100))
filtered_img = median_filter(img, kernel_size=3)