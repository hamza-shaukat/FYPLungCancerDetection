# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 17:35:22 2023

@author: Hamza
"""
import os
import glob
import cv2
import numpy as np

folder = 'C:/Users/Hamza/Desktop/dataset250gs'
images = [cv2.imread(file) for file in glob.glob(folder+"/*/*.jpg")]


z=np.array(images)

np.save('C:/Users/Hamza/Desktop/experiments/imgs250.npy', z)


#image_features = []
label_features=[]
total_images=len(folder)
for i,image_path in enumerate(folder):
    ir_=os.path.basename(os.path.dirname(image_path))
    #image = cv2.imread(image_path)
    #image1=image[...,2]
    #imagea=np.expand_dims(image1,-1)
    #fd, hog_image = hog(image1, orientations=9, pixels_per_cell=(8, 8), 
                        #cells_per_block=(2, 2), visualize=True, multichannel=False)
    
    #plt.imshow(hog_image,cmap='gray')
    #break
    #features = hog.compute(image)
    #image_features.append(fd)
    label_features.append(ir_)
    #print(i+1, '/' , total_images,'-->',round((i+1)/total_images*100,4),'%')
#X=np.array(image_features)
y=np.array(label_features)
#np.save('C:/Users/Hamza/Desktop/saved data/xt.npy', X)
np.save('C:/Users/Hamza/Desktop/experiments/imglbls250.npy', y)