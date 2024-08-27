# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 14:34:43 2023

@author: Hamza
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 13:08:16 2023

@author: Hamza
"""
import os
import glob
import cv2
import numpy as np



#retreive data
#1.assign address of the folder to a string  variable

folder='C:/Users/Hamza/Desktop/dataset5kk'
#list to save all the images 
images=[]
#list to save parent folder name for each image
parent_folders = []
#2 read path of all the images using glob and for loop to iterate on images
for file in glob.glob(folder+"/*/*.jpg"):
    # cv2.imread is used to read all the images from the paths it returns numpy array for each image
    img =cv2.imread(file)
    
    #resizing images to 256,256
    img= cv2.resize(img, (256, 256))
    #converting images to grayscale it returns ndarray of 256,256
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #expand dimension to get 256,256,1
    gs1=np.expand_dims(gray, axis=2)
    #to appent all the images in the list to get 100,256,256,1
    images.append(gs1)
    #to get the parent folder name for each image in a list
    parent_folder =int( os.path.basename(os.path.dirname(file)))
    # append the parent folder name to the list
    parent_folders.append([parent_folder])
#this list images is converted into np array and saved into a var x   
x=np.array(images)
y=np.array(parent_folders)

#y=np.expand_dims(z,axis=1)



np.save('C:/Users/Hamza/Desktop/saved_dataset/newimage5k.npy',x)
np.save('C:/Users/Hamza/Desktop/saved_dataset/newlabel5k.npy',y)






'''import numpy as np
(c,d)=np.load('C:/Users/Hamza/Desktop/experiments/imglbls250.npy')
c.shape
'''





























