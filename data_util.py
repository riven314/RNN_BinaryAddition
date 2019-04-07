#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 11:56:03 2019

Functions:
    1. Reading in data and transform into matrix representation, with function to flip data upside down
    2. Partition data into training set and test set
    3. Shuffle the data
    4. Get randomized minibatches from training data

To do added:
    1. invert the data representation (upside down)

@author: Alex Lau
"""
import numpy as np

# a, b, c * 5000
FILENAME = 'data.txt'

def txt_2_dict(filename):
    """
    retrieve data from A, B, C
    A + B = c 
    
    input:
        file -- str of the filename (must be in same dir) e.g 'data.txt'
    
    output:
        data_dict -- dict of a_list, b_list and c_list
    """
    a_list, b_list, c_list = [], [], []
    with open(filename, "r") as file:
        filein = file.read().splitlines()
        for item in filein:
            tmp_list = item.strip().split(",")
            a_list.append(tmp_list[0])
            b_list.append(tmp_list[1])
            c_list.append(tmp_list[2])
    a_list = str_2_list(a_list)
    b_list = str_2_list(b_list)
    c_list = str_2_list(c_list)
    
    data_dict = dict()
    data_dict['a'] = a_list
    data_dict['b'] = b_list
    data_dict['c'] = c_list
    return data_dict

def str_2_list(data_list):
    """
    txt file datatype conversion
    
    input:
        data_list -- binary str
        
    output:
        ret_list -- list of binary number
    """
    ret_list = []
    for i in range(len(data_list)):
        tmp_list = data_list[i].strip().split(" ")
        tmp_ret_list = [int(tmp_list[0][1]),int(tmp_list[1]),int(tmp_list[2]),int(tmp_list[3]),int(tmp_list[4]),int(tmp_list[5]),int(tmp_list[6]),int(tmp_list[7][0])]
        ret_list.append(tmp_ret_list)
    return ret_list

def dict_2_nparr(data_dict, is_flip = False):
    """
    convert data_dict from txt_2_dict to input data and label data x, y. 
    in the middle, a array and b array are concatenated and go through proper transform for desired shape
    
    input:
        data_dict -- dictionary storing a_list, b_list, c_list, fr
        
    output:
        x -- np.array input data of shape (m, time_steps, input_dim)
        y -- np.array label data of shape (m, time_steps)
    """
    # load in a_list, b_list, c_list
    # use np.copy() to avoid change in place for data_dict
    a_arr = np.array(data_dict['a'], dtype = np.uint8).copy()
    b_arr = np.array(data_dict['b'], dtype = np.uint8).copy()
    y = np.array(data_dict['c'], dtype = np.uint8).copy()
    # parameters for dimension    
    m, time_steps = y.shape
    input_dim = 2
    # Transform data into np array x, y
    ab_arr = np.c_[a_arr, b_arr]
    x = np.array(ab_arr).reshape([m, input_dim, time_steps])
    # convert from shape (x, y, z) to shape (x, z, y)
    x = np.transpose(x, (0, 2, 1))
    # flip x, y if specified
    if is_flip:
        x = np.flip(x, 1)
        y = np.flip(y, 1)
        print('x and y are flipped upsided down')
    print('x shape = {}'.format(x.shape))
    print('y shape = {}'.format(y.shape))
    return x, y 

def partition_data(x, y, train_idx = 0, num4train = 4000, test_idx = 4000, num4test = 1000):
    """
    partition data into trainnig set and test set
    training set - x[train_idx:train_idx+num4train], y[test_idx:test_idx+num4test]
    test set - x[test_idx:test_idx+num4test], y[test_idx:test_idx+num4test]
    
    input:
        x -- np.array input data of shape (m, time_steps, input_dim)
        y -- np.array label data of shape (m, time_steps)
        trian_idx -- idx at dim(m) timewhere you start pick the training data
        num4train -- number of trainig input data
        test_idx -- idx at dim(m) timewhere you start pick the test data
        num4test -- number of test input data
        
    output:
        train_x -- np.array of shape (batch_size, time_steps, input_dim)
        train_y -- np.array of shape (batch_size, time_steps)
        test_x -- np.array of shape (batch_size, time_steps, input_dim)
        test_y -- np.array of shape (batch_size, time_steps)
    """    
    train_x = x[train_idx: train_idx + num4train, :, :]
    train_y = y[train_idx: train_idx + num4train, :]
    test_x = x[test_idx: test_idx + num4test, :, :]
    test_y = y[test_idx: test_idx + num4test, :]
    print('train_x shape = {}'.format(train_x.shape))
    print('train_y shape = {}'.format(train_y.shape))
    print('test_x shape = {}'.format(test_x.shape))
    print('test_y shape = {}'.format(test_y.shape))
    return train_x, train_y, test_x, test_y

def shuffle(x, y, rand_seed = 1):
    """
    shuffle input data and label data in place (i.e rearrange the order)
    
    input:
        x -- before shuffle np.array input data of shape (m, time_steps, input_dim)
        y -- before shuffle np.array label data of shape (m, time_steps)
    
    output:
        x -- after shuffle np.array input data of shape (m, time_steps, input_dim)
        y -- after shuffle before shuffle np.array label data of shape (m, time_steps)
    """
    m, _, _ = x.shape
    shuff_idx = np.random.permutation(m)
    x = x[shuff_idx, :, :]
    y = y[shuff_idx, :]
    return x, y

def get_minibatches(x, y, it, bs_size):
    """
    get a minibatch of size bs_size from the data set x and y. Take care of end case where minibatch may have size less than bs_size
    ** make sure the sample size for training is divisible by bs_size, the function cant take care of the last minibatch with sample size less than bs_size
    
    input:
        x -- after shuffle np.array input data of shape (m, time_steps, input_dim)
        y -- after shuffle np.array label data of shape (m, time_steps)
        it -- number of epoch (start from 0). if it is the last, minibatch size is <= bs_size
        bs_size -- minibatch size. it is the upper bound of the minibatch
        
        
    output:
        mb_x -- minibatch for x of shape (<=bs_size, time_steps, input_dim)
        mb_y -- minibatch for y of shape (<=bs_Size, time_steps)
    """
    m, _, _ = x.shape
    bs_num = m // bs_size
    if it is not bs_num:
        mb_x = x[it * bs_size: (it+1) * bs_size, :, :]
        mb_y = y[it * bs_size: (it+1) * bs_size, :]
    else:
        mb_x = x[it * bs_size:, :, :]
        mb_y = y[it * bs_size:, :]
    return mb_x, mb_y

