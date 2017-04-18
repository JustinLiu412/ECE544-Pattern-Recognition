
# coding: utf-8
# Codes from team formed by Junze (Justin) Liu and Changsong Dong

# In[1]:

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# In[11]:

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

# In[]:

def label_edit(label):
    """
    Edit label and change the domain of it from {0, 1} to {-1, 1}
    
    """
    
    temp = label
    for i in range(len(label)):
        if temp[i][0] == 0:
            temp[i][0] = -1
            
    return temp

# In[12]:

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


# In[14]:

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


# In[15]:

def linear_regression_gradient(data, label, weight, b):
    """
    Calculate the gradient of linear node classifier. Return the gradient.
    learning rate: 1e-5 or 1e-6
    """

    gradient_w, gradient_b = 0, 0 #Initialize the gradient of w and b
    
    for i in range(len(label)):
        gradient_w -= (-2) * (label[i] - (np.dot(weight, data[i]) + b)) * data[i]
        gradient_b += (-2) * (label[i] - (np.dot(weight, data[i]) + b))

    return gradient_w, gradient_b

# In[]:
def logistic_regression_gradient(data, label, weight, b):
    """Calculate the gradient of logistic regression . Return the gradient
    
    """
    
    gradient_w, gradient_b = 0, 0
    for i in range(len(label)):
        gradient_w += (-2) * ((np.dot(weight, data[i]) + b) - label[i]) * (np.dot(weight, data[i]) + b) * \
                   (1 - (np.dot(weight, data[i]) + b)) * data[i]
        gradient_b += (-2) * ((np.dot(weight, data[i]) + b) - label[i]) * (np.dot(weight, data[i]) + b) * \
                   (1 - (np.dot(weight, data[i]) + b))
        
    return gradient_w / len(label), gradient_b / len(label)

# In[]:

def perceptron_gradient(data, label, weight, b = 0):
    """
    Calculate the gradient of perceptron classifier. Return the gradient.
    learning rate: 1e-2
    """
    
    gradient_w = 0 #Initialize the gradient of weight

    for i in range(len(label)):
        if np.dot(weight, data[i]) * label[i] < 0 :
            gradient_w += (-1) * data[i] * label[i]
        else:
            gradient_w += 0
    
    gradient_w = gradient_w / len(label)
    
    return gradient_w, b
    
#data, label = get_data('dev')
#weight = 2 * np.random.random(size = 16) - 1
#for i in range(len(label)):
#    print label[i, 0]
#    print np.dot(weight, data[i])

# In[]:
#
#def svm_gradient(C, data, label, w, b = 0):
#    """
#    Calculate the gradient of svm classifier. Return the gradient.
#    
#    """
#    
#    gradient_w = 0
#    gradient_b = 0
#    
#    for i in range(len(label)):
##        if label[i] != np.sign(np.dot(w, data[i]) * label[i]) :
#        if np.dot(w, data[i]) * label[i] < 0 :
#            gradient_w += C * (-1) * data[i] * label[i]
#            gradient_b += C * (-1) * label[i]
#        else:
#            gradient_w += 0
#            gradient_b += 0
#        #print gradient_w
#            
#    gradient_w = (2 * w + gradient_w) / len(label)
#    
#    #gradient_w = gradient_w / len(label)    
#    
#    return gradient_w, gradient_b    
def svm_gradient(C, data, label, w, b = 0):
    """
    Calculate the gradient of svm classifier. Return the gradient.
    
    """
    
    gradient_w = 0
    gradient_b = 0
    
    for i in range(len(label)):
        if label[i] * np.dot(w, data[i]) < 1 :
        # label[i] != np.sign(np.dot(w, data[i]) * label[i]) :
            gradient_w += C * (-1) * data[i] * label[i]
            gradient_b += C * (-1) * label[i]
        else:
            gradient_w += 0
            gradient_b += 0
        #print gradient_w
            
    gradient_w = (2 * w + gradient_w) #/ len(label) 
    
    return gradient_w, gradient_b
        
# In[16]:

def gradient_descent(weight, b, learning_rate, gradient_w = 0, gradient_b = 0):
    """Update and return weight and b.
    
    """
    
    weight -= learning_rate * gradient_w
    b -= learning_rate * gradient_b

    return weight, b

# In[18]:

def compute_mse(data, label, w, b):
    """Compute the mean square error
    
    """
    
    mse = 0
    
    for i in range(len(label)):
        mse += (label[i] - (np.dot(w, data[i]) + b)) ** 2
        
    mse = mse / len(label)
    
    return mse

# In[]:

def compute_acc(data, label, w, b):
    """Compute the accuracy
    
    """
    
    acc = 0
    for i in range(len(label)):
        #if label[i] == round(np.dot(w, data[i])+b):
        if label[i] == np.sign(np.dot(w, data[i]) + b):
            acc += 1
    return acc / float(len(label))
    
# In[]:
#    
#def PCA(data, label):
#    """
#    """
#    
#    m_x = np.sum(data, axis = 0) / len(data)
#    X_T = np.transpose(data - m_x)
#    X = data - m_x
#    E = np.dot(X, X_T)
#    E.linalg.eigvals(a)
#    
    
# In[19]:

def activate(epoch = 4000):
    """
    
    """

    # data and parameter initialization
 #   w = 2 * np.random.random(size = 16) - 1
    w_li = 2 * np.random.random(size = 2) - 1
    w_lo = 2 * np.random.random(size = 2) - 1
    w_per = 2 * np.random.random(size = 2) - 1
    w_svm = 2 * np.random.random(size = 2) - 1
    b = 0
    b_li = 0
    b_lo = 0
    b_per = 0
    b_svm = 0
    lr_li = 0.0000001
    lr_lo = 0.00001
    lr_per = 0.01
    lr_svm = 0.0001
#    accuracy = np.zeros(epoch)
#    error_rate = np.zeros(epoch)
#    train_accuracy_li = np.zeros(epoch)
    train_error_rate_li = np.zeros(epoch)
#    train_accuracy_lo = np.zeros(epoch)
    train_error_rate_lo = np.zeros(epoch)
#    train_accuracy_per = np.zeros(epoch)
    train_error_rate_per = np.zeros(epoch)
#    train_accuracy_svm = np.zeros(epoch)
    train_error_rate_svm = np.zeros(epoch)
#    iteration = np.linspace(0, epoch-1, epoch, endpoint = True)

#    train_data, train_label = get_data('train') #build the dataset for training network
#    eval_data, eval_label = get_data('eval')
#    train_label_11 = label_edit(train_label)
#    eval_label_11 = label_edit(eval_label)
    
    pca_data, pca_label = get_data('train')
    pca_data_11, pca_label_11 = get_data('train')
    pca_label_11 = label_edit(pca_label_11)
    pca_data, pca_label = shuffle(pca_data, pca_label)
    pca = PCA(n_components = 2).fit(pca_data)
    pca_data = pca_data[0:300][:]
    pca_label = pca_label[0:300][:]
    pca_label_11 = pca_label_11[0:300][:]
    pca_data = pca.fit_transform(pca_data)
    
    
    for i in range(epoch):    
        g_w_li, g_b_li = linear_regression_gradient(pca_data, pca_label, w_li, b_li)
        g_w_lo, g_b_lo = logistic_regression_gradient(pca_data, pca_label, w_lo, b_lo)
        g_w_per, g_b_per = perceptron_gradient(pca_data, pca_label_11, w_per, b_per)
        g_w_svm, g_b_svm = svm_gradient(0.1, pca_data, pca_label_11, w_svm, b_svm)#perceptron
        
        w_li, b_li = gradient_descent(w_li, b, lr_li, g_w_li, g_b_li)
        w_lo, b_lo = gradient_descent(w_lo, b, lr_lo, g_w_lo, g_b_lo)
        w_per, b_li_per = gradient_descent(w_per, b, lr_per, g_w_per, g_b_per)
        w_svm, b_svm = gradient_descent(w_svm, b, lr_svm, g_w_svm, g_b_svm)

#        mse = compute_mse(dev_data, dev_label, w, b)
#        accuracy[i] = compute_acc(eval_data, eval_label, w, b)
#        error_rate[i] = 1 - accuracy[i]
        
        train_error_rate_li[i] = 1 - compute_acc(pca_data, pca_label, w_li, b)
        train_error_rate_lo[i] = 1 - compute_acc(pca_data, pca_label, w_lo, b)
        train_error_rate_per[i] = 1 - compute_acc(pca_data, pca_label_11, w_per, b)
        train_error_rate_svm[i] = 1 - compute_acc(pca_data, pca_label_11, w_svm, b)


    red_x, red_y = [], []
    blue_x, blue_y = [], []

    for i in range(len(pca_data)):
        if pca_label[i] == 0:
            red_x.append(pca_data[i][0])
            red_y.append(pca_data[i][1])
        elif pca_label[i] == 1:
            blue_x.append(pca_data[i][0])
            blue_y.append(pca_data[i][1])
            
        k_li = w_li[1] / w_li[0]
        k_lo = w_lo[1] / w_lo[0]       
        k_per = w_per[1] / w_per[0]
        k_svm = w_svm[1] / w_svm[0]

#    print np.shape(red_x), np.shape(red_y)#, pca_label

    x = np.linspace(-50, 50, 256, endpoint=True)
    plt.figure(figsize=(10,6), dpi=200)
    plt.scatter(red_x, red_y, color='red', marker='x')
    plt.scatter(blue_x, blue_y, color='blue', marker='o')
    plt.plot(x, k_li * x + b_li, color = 'purple', label = 'Linear')
    plt.plot(x, k_lo * x + b_lo, color = 'blue', label = 'Logistic')
    plt.plot(x, k_per * x + b_per, color = 'green', label = 'Perceptron')
    plt.plot(x, k_svm * x + b_svm, color = 'red', label = 'SVM')
    plt.xlim(-30, 15)
    plt.ylim(-6, 6)
    plt.legend(loc='upper left', frameon=False)
    plt.title('Iteration: 2500')
    
    
    plt.show()

#        print train_error_rate_per[i]
    print 'li: ',train_error_rate_li[2499], ' lo:',train_error_rate_lo[2499], \
          'per:', train_error_rate_per[2499], 'svm:', train_error_rate_svm[2499]
#    plt.xlim(0.0,epoch)
#    plt.ylim(0.0, 1)
#    
#    plt.figure(figsize=(20,12), dpi=200)
#    plt.subplot(2,2,1)
#    plt.title('Linear Regression')
#    plt.plot(iteration, train_error_rate_li, linewidth=2.5, linestyle="-")
#    plt.subplot(2,2,2)
#    plt.title('Logistic Regression')
#    plt.plot(iteration, train_error_rate_lo, linewidth=2.5, linestyle="-")
#    plt.subplot(2,2,3)
#    plt.title('Perceptron')
#    plt.plot(iteration, train_error_rate_per, linewidth=2.5, linestyle="-")
#    plt.subplot(2,2,4)
#    plt.title('SVM')
#    plt.plot(iteration, train_error_rate_svm, linewidth=2.5, linestyle="-")
#    plt.subplot(2,1,2)
#    plt.plot(iteration, error_rate)

        
#        print accuracy
#        print error_rate
        
        #if i % 1000 == 0:
        #print ("epoch: %d, error rate: %f." % (i+1, error_rate))
        #print mse
        #print g_w
        #print dev_label
        #for i in range(len(label)):
        #    print dev_label[i]
        #    if i == 100:
        #        print np.dot(w, data[i])+b
        #    print label[i] == np.sign(np.dot(w, data[i]) + b)
        #    print round(np.dot(w, data[i])+b)
        #    print dev_label[i] == np.sign(np.dot(w, dev_data[i]) * dev_label[i])

# In[20]:

activate()

