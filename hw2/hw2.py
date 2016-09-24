# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 10:16:02 2016

@author: changsongdong
"""

import numpy as np
import random

def nan_check(data, label):
    """Find out the nan-rows in datasets and delete these rows
    
    """
    
    nan_rows = np.array(0); #define an array containg the no. of rows having 'nan'
    
    #collect all the numbers of 'nan'-data rows
    for i in range(len(data)):
        for j in range(16):
            if str(data[i][j]) == 'nan':
                nan_rows = np.append(nan_rows, i)
    nan_rows = np.delete(nan_rows, 0) #delete the first element of nan_rows which was made to fit the append()
    
    #output the dataset whose 'nan'-data rows have been deleted
    return np.delete(data, nan_rows, 0), np.delete(label, nan_rows, 0)
    
def label_edit(label):
    """
    Edit label and change the domain of it from {0, 1} to {-1, 1}
    
    """
    
    temp = label
    for i in range(len(label)):
        if temp[i][0] == 0:
            temp[i][0] = -1
            
    return temp
    
def shuffle(data_set, label_set):
    """Randomly shuffle the data and label
    
    data_set    the data samples
    
    label_set   the lables
    """
    
    shuffled_data = np.zeros((data_set.shape))
    shuffled_label = np.zeros((label_set.shape))
    idx = np.array(xrange(len(label_set)))
    random.shuffle(idx)
    i = 0
    for j in idx:
        shuffled_data[i] = data_set[int(j)]
        shuffled_label[i] = label_set[int(j)]
        i += 1
    return shuffled_data, shuffled_label