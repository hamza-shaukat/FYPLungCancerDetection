# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 17:54:13 2023

@author: Hamza
"""






import os
import glob
import cv2
import numpy as np

folder = 'C:/Users/Hamza/Desktop/dataset50gs'








a=len(os.listdir(folder))
for i,(filename) in enumerate(os.listdir(folder)):
    # Load the image
    img = cv2.imread(f'{folder}/{filename}')
    #print(type(img),img.shape)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(type(gray),gray.shape)
    gs1=np.expand_dims(gray, axis=2)
    #print(type(gs1),gs1.shape)
    #break
    # Save the grayscale image
    cv2.imwrite(f'{folder}/{filename}', gray)
    print(i+1,'/',a)
    
images = [cv2.imread(file) for file in glob.glob(folder+"/*/*.jpg")]   
z=np.array(images)