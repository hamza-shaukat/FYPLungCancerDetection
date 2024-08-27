# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 13:55:16 2023

@author: Hamza
"""

import os
import glob

# folder containing the images
folder1 = "C:/Users/Hamza/Desktop/dataset250/1"
#folder2 = "path/to/folder2"

# use glob to get a list of all the image files in the folder
image_files1 = glob.glob(os.path.join(folder1, "*.jpg"))
#image_files2 = glob.glob(os.path.join(folder2, "*.jpg"))

# concatenate the two lists of image files
image_files = image_files1# + image_files2

# create an empty list to store the parent folder names
parent_folders = []

# loop through the image files and get the parent folder name
for image_file in image_files:
    # get the parent folder name
    parent_folder = os.path.basename(os.path.dirname(image_file))
    # append the parent folder name to the list
    parent_folders.append(parent_folder)

# print the list of parent folder names
print(parent_folders)