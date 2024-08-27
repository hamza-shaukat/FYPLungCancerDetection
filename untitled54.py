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

folder = 'C:/Users/Hamza/Desktop/dataset50'
images = [cv2.imread(file) for file in glob.glob(folder+"/*/*.jpg")]


x=np.array(images)

np.save('C:/Users/Hamza/Desktop/experiments/a.npy', x)


#image_features = []
label_features=[]
total_images=len(folder)
for i,image_path in enumerate(folder):
    ir_=os.path.basename(os.path.dirname(image_path))


    label_features.append(ir_)
   # print(i+1, '/' , total_images,'-->',round((i+1)/total_images*100,4),'%')

y=np.array(label_features)

np.save('C:/Users/Hamza/Desktop/experiments/aa.npy', y)