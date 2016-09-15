
# coding: utf-8

# In[55]:

import numpy as np
import random
import os

def get_data(set_type):
    """Get data from files and storage them in a array. Return the data_set and label_set.
    
    set_type    the type of data set you want to build, including train dataset, dev dataset 
                and eval dataset
    """
    
    data_path = {'train': 'train/lab/hw1train_labels.txt', 'dev': 'dev/lab/hw1dev_labels.txt', 'eval': 'eval/lab/hw1eval_labels.txt'} #find the label file path needed for building dataset
    path_prefix = '/Users/justinliu/Documents/ECE544-Patterns Recognization/hw1/'
    full_path = path_prefix + data_path[set_type] #find the full path of the label file
    label_array = np.loadtxt(full_path, dtype='string') #load the label file into a array

    #creat empty arrays to obtain label and data
    label_set = np.zeros(len(label_array))
    data_set = np.zeros([len(label_array), 16])
    #the first column of the label file is for labels,
    #the second column is the corresbonding data file nam
    for i in range(len(label_array)): 
        #build the label set
        label_set[i] = label_array[i][0] #put labels into label_set
        #build the data set
        with open(path_prefix + label_array[i][1]) as data_file:
            data = data_file.readlines()[0].split() #find the data accoding to label
        for j in range(len(data)):
            data_set[i][j] = data[j] #put data into the dataset

    data_set, label_set, nan_rows = nan_check(data_set, label_set) #delete the rows containing 'nan'
    
    return data_set, label_set#, nan_rows #return the data set and label set

#train_data, train_label, nan_rows = get_data('train')
#print np.shape(train_label), np.shape(train_data)
#print nan_rows

#print train_label


# In[46]:

def nan_check(data, label):
    """Find out the rows in datasets and delete these rows
    
    """
    
    nan_rows = np.array([0]); #define an array containg the no. of rows having 'nan'
    
    #collect all the numbers of 'nan'-data rows
    for i in range(len(data)):
            if str(data[i][1]) == 'nan':
                nan_rows = np.append(nan_rows, i)
    
    nan_rows = np.delete(nan_rows, 0) #delete the first element of nan_rows which was made to fit the append()
    
    return np.delete(data, nan_rows, 0), np.delete(label, nan_rows), nan_rows #output the dataset whose 'nan'-data rows have been deleted
                


# In[57]:

def shuffle(data_set, label_set):
    """Randomly shuffle the data and label
    
    data_set    the data samples
    
    label_set   the lables
    """
    
    shuffled_data = np.zeros((data_set.shape))
    shuffled_label = np.zeros((len(label_set)))
    idx = np.array(xrange(len(label_set)))
    random.shuffle(idx)
    # i = 0
    # for j in idx:
    #     shuffled_data[i] = data_set[int(j)]
    #     shuffled_label[i] = label_set[int(j)]
    #     i += 1
    
    return shuffled_data, shuffled_label

#train_data, train_label = shuffle(train_data, train_label)
#print train_label


# In[58]:

def linear_node_gradient(data, label, weight, b, gradient_w, gradient_b):
    """Calculate the gradient of linear node classifier. Return the gradient.
    
    """

    for i in range(len(label)):
        gradient_w += (-2) * (label[i] - (np.dot(weight, data[i]) + b)) * data[i]
        gradient_b += (-2) * (label[i] - (np.dot(weight, data[i]) + b))
        #print gradient_w and gradient_b
    return gradient_w, gradient_b

#w = 2 * np.random.random(size = 16) - 1
#print w
#b = 0
#print w
#print train_data[1]
#g_w = linear_node_gradient(train_data,train_label, w, b, 0, 0)
#print g_w
#g_w, g_b = linear_node_gradient(train_data, train_label, w, b)


# In[59]:

def logistic_regression_gradient(data, label, weight,b, gradient_w, gradient_b):
    """Calculate the gradient of logistic regression . Return the gradient
    
    """
    
    for i in range(label):
        gradient_w += (-2) * (label[i] - (np.dot(weight, data[i]) + b)) * (np.dot(weight, data[i])        + b) * (1 - (np.dot(weight, data[i]) + b)) * data[i]
        gradient_b += (-2) * (label[i] - (np.dot(weight, data[i]) + b)) * (np.dot(weight, data[i])        + b) * (1 - (np.dot(weight, data[i]) + b))
        
    return gradient_w, gradient_b

# In[61]:

def gradient_descent(weight, bias, learning_rate, gradient_w, gradient_b):
    """Update and return weight and b.
    
    """
    
    weight = weight - learning_rate * gradient_w
    bias = bias - learning_rate * gradient_b
    return weight, bias


# In[52]:

def compute_MSE(data, label, weight, b, mse):
    """Compute the Mean Square Error
    
    """
    

    for i in range(len(label)):
        mse += (label[i] - (np.dot(weight, data[i])+b)) ** 2
        
    mse = mse / len(label)
    return mse
   


# In[56]:

def activate_train(lr = 0.01, phase = 'train'):
    """
    
    phase:    can be train/dev/eval
    """
    
    # data and parameter initialization
    train_data, train_label = get_data(phase) #build the dataset for training network
    w = 2 * np.random.random(size = 16) - 1
    b = 0
    g_w, g_b = 0, 0

    # train the model
    train_data, train_label = shuffle(train_data, train_label) #shuffle the dataset
    g_w, g_b = linear_node_gradient(train_data, train_label, w, b, g_w, g_b) #choose the classifier
    w, b = gradient_descent(w, b, lr, g_w, g_b)
    
    return w, b
    

# In[ ]:

def activate_dev(phase = 'dev'):
    """
    
    """
    mse = np.array([0.0])
    dev_data, dev_label = get_data(phase)
    w, b = activate_train()
    mse = compute_MSE(dev_data, dev_label, w, b, mse)
    
    return mse
    
# In[ ]:
#dev_data, dev_label = get_data('dev')
#print np.shape(dev_data), np.shape(dev_label)
epoch = 5
for i in range(epoch):
    print activate_train()
    print activate_dev()


# In[ ]:



