# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 22:01:08 2022

@author: Hamza
"""


# import numpy
import numpy as np
  
# using Numpy.expand_dims() method
gfg = np.array([[1, 2], [7, 8]])
print(gfg.shape)
  
gfg = np.expand_dims(gfg, axis = 2)
print(gfg.shape)