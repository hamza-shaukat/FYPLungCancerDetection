# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 21:19:14 2023

@author: Hamza
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 20:04:30 2023

@author: Hamza
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 17:35:22 2023

@author: Hamza
"""
import os
import glob
import cv2
import numpy as np


#retreive data
#1.assign address of the folder to a string  variable
#folder='path'
#2 read path of all the images using glob and for loop to iterate on images
#for file in glob.glob(folder+"/*/*.jpg"):

# cv2.imread is used to read all the images from the paths it returns numpy array for each image
# a list is initialized for saving all the images in it in the form of numpy array
#images =cv2.imread(file)
#this list images is converted into np array and saved into a var x
#x=np.array(images)


































folder = 'C:/Users/Hamza/Desktop/dataset50gs'
#images = [cv2.imread(file) for file in glob.glob(folder+"/*/*.jpg")]
for file in glob.glob(folder+"/*/*.jpg"):
    images =cv2.imread(file)

x=np.array(images)

np.save('C:/Users/Hamza/Desktop/experiments/imgs5kk.npy', x)

class_labels = []
#image_features = []
label_features=[]
total_images=len(folder)
for i,image_path in enumerate(folder):
    ir_=os.path.basename(os.path.dirname(image_path))
    
    
    #class_labels.append(os.path.basename(os.path.dirname(image_path)))

    #print(class_labels)

    label_features.append(ir_)
   # print(i+1, '/' , total_images,'-->',round((i+1)/total_images*100,4),'%')

y=np.array(label_features)

np.save('C:/Users/Hamza/Desktop/experiments/imglbls5kk.npy', y)



'''


for path in ir_:
    class_labels.append(os.path.basename(os.path.dirname(path)))

print(class_labels)'''