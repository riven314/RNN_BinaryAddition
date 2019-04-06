#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 11:59:58 2019

Functions:
    1. get prediction from network output
    2. get accuracy rate from prediction vectors (partial credit is given for inccorrect label)
    3. plot the metrics (accuracy rate and loss against epoches)

@author: Alex Lau
"""
import matplotlib.pyplot as plt
import numpy as np

def get_prediction(probs):
    """
    convert np array of probability from network output to a prediction np array
    
    input:
        probs -- np.array of shape (batch_size, binary_dim, 1), output from network after sigmoid activation
    
    output:
        preds -- np.array of prediction
    """
    bs_size, binary_dim, _ = probs.shape
    tmp = np.array(probs).reshape([bs_size, binary_dim])    
    preds = np.array([np.where(p>=0.5, 1, 0) for p in tmp])
    return preds

def get_accuracy(y, preds):
    """
    get accuracy from actual labels (y) and prediction from model (preds)
    
    input:
        y -- np.array of shape (batch_size, binary_dim), represent batch of actual labels
        preds -- np.array of shape (batch_size, binary_dim), represent prediction by model
        
    output:
        acc -- float of accuracy rate on batch y
    """
    accuracy = 0
    for i in range(len(y)):
        len_ = len(y[i])
        tmp_num = 0
        for j in range(len_):
            if y[i][j] == preds[i][j]:
                tmp_num += 1
        #print('tmp_num = {}'.format(tmp_num))
        accuracy += tmp_num / len_
        #print('accuracy = {}'.format(accuracy))
    accuracy /= len(y)
    return accuracy 

def get_metrics_plot(train_cost_list, test_cost_list, train_acc_list, test_acc_list):
    """
    plot a curve for loss evolution and a plot for accuracy evolution (for both training set and test set)
    
    input:
        train_cost_list -- 
        test_cost_list -- 
        train_acc_list --
        test_acc_list --
        
    output:
        None 
    """
    f, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1)
    # plotting loss curve
    cost_train, = ax1.plot(train_cost_list)
    cost_test, = ax1.plot(test_cost_list)
    ax1.legend((cost_train, cost_test), ('cost for train', 'cost for test'))
    ax1.set_title('Loss Evolution')
    # plotting accuracy curve
    acc_train, = ax2.plot(train_acc_list)
    acc_test, = ax2.plot(test_acc_list)
    ax2.legend((acc_train, acc_test), ('accuracy for train', 'accuracy for test'))
    ax2.set_title('Accuracy Evolution')
    plt.tight_layout()
    return None