
# coding: utf-8

# In[141]:

import numpy as np
import random



# In[142]:

def nan_check(data, label):
    """Find out the rows in datasets and delete these rows
    
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


# In[143]:

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


# In[144]:

def get_data(set_type):
    """Get data from files and storage them in a array. Return the data_set and label_set.
    
    set_type    the type of data set you want to build, including train dataset, dev dataset 
                and eval dataset
    """
    
    data_path = {'train': 'train/lab/hw1train_labels.txt', 'dev': 'dev/lab/hw1dev_labels.txt',                  'eval': 'eval/lab/hw1eval_labels.txt'} 

    label_array = np.loadtxt(data_path[set_type], dtype='string') #load the label file into a array

    #creat empty arrays to insert label and data
    label_set = np.zeros([len(label_array), 1])
    data_set = np.zeros([len(label_array), 16])
    
    # the first column of the label file is the label,
    # the second column is the corresbonding data file nam
    for i in range(len(label_array)): 
        #build the label set
        label_set[i] = label_array[i][0] # insert label into label_set
        
        #build the data set
        with open(label_array[i][1]) as data_file:
            data = data_file.readlines()[0].split() #find the data accoding to label
        for j in range(len(data)):
            data_set[i][j] = data[j] #insert data into the dataset
            
    data_set, label_set = nan_check(data_set, label_set) #delete the rows containing 'nan'

    return shuffle(data_set, label_set) #return the shuffled data set and label set


# In[145]:

def linear_regression_gradient(data, label, weight, b):
    """Calculate the gradient of linear node classifier. Return the gradient.
    
    """

    gradient_w, gradient_b = 0, 0
    for i in range(len(label)):
        gradient_w += (-2) * (label[i] - (np.dot(weight, data[i]) + b)) * data[i]
        gradient_b += (-2) * (label[i] - (np.dot(weight, data[i]) + b))

    return gradient_w / len(label), gradient_b / len(label)


# In[146]:

def gradient_descent(weight, b, learning_rate, gradient_w, gradient_b):
    """Update and return weight and b.
    
    """
    
    weight -= learning_rate * gradient_w
    b -= learning_rate * gradient_b
    return weight, b


# In[147]:

def compute_MSE(data, label, weight, b, mse):
    """Compute the Mean Square Error
    
    """
    
    for i in range(len(label)):
        mse += (label[i] - (np.dot(weight, data[i]) + b)) ** 2
        
    mse = mse / len(label)
    return mse


# In[148]:

def compute_mse(dev_data, dev_label, w, b):
    """Compute the mean square error
    
    """
    
    mse = 0
    mse = compute_MSE(dev_data, dev_label, w, b, mse)
    
    return mse

def compute_acc(data, label, w, b):
    """accuracy
    
    """
    
    acc = 0
    for i in range(len(label)):
        if label[i] == round(np.dot(w, data[i]) + b):
            acc += 1
    return acc / float(len(label))
# In[157]:
def activate(epoch = 2000, lr = 0.01):
    """
    
    """

    # data and parameter initialization
    w = 2 * np.random.random(size = 16) - 1
    b = 0

    train_data, train_label = get_data('train') #build the dataset for training network
    dev_data, dev_label = get_data('dev')
    
    for i in range(epoch):    
        g_w, g_b = linear_regression_gradient(train_data, train_label, w, b)
        w, b = gradient_descent(w, b, lr, g_w, g_b)
    
        mse = compute_mse(dev_data, dev_label, w, b)
        acc = compute_acc(dev_data, dev_label, w, b)
        
        print("epoch %d, loss: %f, error rate: %s " % (i, mse, 1 - acc))
# In[158]:

activate()


# In[ ]:



