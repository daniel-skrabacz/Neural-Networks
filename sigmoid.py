#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 12:36:06 2021

@author: dskrabacz
"""
from numpy import exp
L = [1, 2, 3]

my_list = [exp(j) for j in L]

def softmax(L):
    my_list = [exp(j) for j in L]
    my_sum = sum(my_list)
    my_moid = [exp(i)/my_sum for i in L]
    return my_moid
        
#%% Cross Entropy
import numpy as np
import math
Y = [ 1,0,1,1]
P = [0.4,0.6,0.1,0.5]

my_list = []
for i,j in zip(np.float_(Y),np.float_(P)):
    print(i,j)
    answer = i*np.log(j)+(1-i)*np.log((1-j))
    my_list.append(answer)
    
print(np.sum(my_list))

#%% Gradient Descent
import numpy as np
def sigmoid(x):
    return 1/ (1+np.exp(-x))