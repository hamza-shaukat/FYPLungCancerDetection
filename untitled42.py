# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:01:51 2023

@author: Hamza
"""

import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
#lib for lazy
from lazypredict.Supervised import LazyClassifier
#from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
#-------

folder_path = 'C:/Users/Hamza/Desktop/aaaa'

folder_images = glob(folder_path + '/*/*.jpg')





total_images=len(folder_images)
for i,image_path in enumerate(folder_images):
    igray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)