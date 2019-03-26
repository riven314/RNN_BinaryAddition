#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:50:03 2019

@author: Alex Lau

Guiding Question:
1. how to initialise the weight? (Xavier initialiser)
2. using what learning rate and optimiser?
3. early stop?
4. grid search for hyper parameters?
5. use of LSTM/ GRU?
6. use minibatch? (shuffle)
7. flexible weight?
8. using different activation function in hidden unit?

Issue:
1. how to keep track fo the weight? 
2. tf.reduce_sum ?? 
3. tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim, activation= **tf.nn.sigmoid) ??
4. loss function in test set inconsistent with training set?
5. accuracy allows partial credit to wrong ans

Further:
1. use tensorboard to keep track of key metrics
2. Plot loss function

"""
import numpy as np
import tensorflow as tf

# a, b, c * 5000
FILENAME = 'data.txt'

def txt_2_data(filename):
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

def get_test_acc(remain_result):
    """
    check accuracy for test result
    
    input:
        remain_result -- np array of remaining result
    
    output:
        accuracy -- float of accuracy
    """
    accuracy = 0
    for i in range(len(remain_result)):
        len_ = len(remain_result[i][0])
        tmp_num = 0
        for j in range(len_):
            if remain_result[i][0][j] == remain_result[i][1][j]:
                tmp_num += 1
        accuracy += tmp_num / len_
    print('Accuracy on remain result = {}'.format(accuracy))
    return accuracy    

def BasicRNN_Setup(hidden_dim, lr, optimiser = 'adam'):
    """
    set up the weight, placeholder and model architecture for implementating basic RNN
    
    input:
        hidden_dim -- dimension of hidden unit
        lr -- learning rate
        optimiser -- str indicating type of optimiser. default to be adam (adam/ rmsprop/ nesterov_momentum/ momentum)
    output:
        train_step -- ready for training
        cache -- dictionary of tensor, including
            weights -- W, b, h, h_
            variables -- X, Y, Y_
            model -- model object
            loss 
    """
    # BASIC PARAMETERS
    time_steps = 8        # time steps which is the same as the length of the bit-string
    input_dim = 2         # number of units in the input layer
    hidden_dim = 4       # number of units in the hidden layer
    output_dim = 1        # number of units in the output layer
    binary_dim = 8
    largest_number = pow(2, binary_dim)
    
    # DEFINE PLACEHOLDER X AND Y
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, time_steps, input_dim], name='x')
    Y = tf.placeholder(tf.float32, [None, time_steps], name='y')
    
    # DEFINE MODEL OBJECT
    cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim, activation=tf.nn.sigmoid)
    
    # DEFINE WEIGHTS
    # values is a tensor of shape [batch_size, time_steps, hidden_dim]
    # last_state is a tensor of shape [batch_size, hidden_dim]
    values, last_state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    values = tf.reshape(values,[time_steps, hidden_dim])
    # put the values from the RNN through fully-connected layer
    W = tf.Variable(tf.random_uniform([hidden_dim, output_dim], minval=-1.0,maxval=1.0), name='W')
    b = tf.Variable(tf.zeros([1, output_dim]), name='b')
    h = tf.nn.sigmoid(tf.matmul(values,W) + b, name='h')

    # DEFINE MODEL    
    # minimize loss, using ADAM as weight update rule
    h_ = tf.reshape(h, [time_steps])
    Y_ = tf.reshape(Y, [time_steps])
    
    # DEFINE LOSS
    loss = tf.reduce_sum(-Y_ * tf.log(h_) - (1-Y_) * tf.log(1-h_), name='loss')
    
    # DEFINE OPTIMISER AND LEARNING RATE
    if optimiser == 'adam':
        train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    elif optimiser == 'rmsprop':
        train_step = tf.train.RMSPropOptimizer(lr).minimize(loss)
    elif optimiser == 'nesterov_momentum':
        train_step = tf.train.MomentumOptimizer(learning_rate = lr, use_nesterov = True).minimise(loss)
    elif optimiser == 'momentum':
        train_step = tf.train.MomentumOptimizer(learning_rate = lr, use_nesterov = False).minimise(loss)
    
    # STORE OUTPUT
    cache = dict()
    cache['W'] = W
    cache['b'] = b
    cache['h'] = h
    cache['h_'] = h_
    cache['X'] = X
    cache['Y'] = Y
    cache['Y_'] = Y_
    cache['cell'] = cell
    cache['loss'] = loss
    
    return train_step, cache
    

def train_model(data_dict, train_step, cache, n_epoch, num4train):
    """
    feed in data and train model
    cost function is binary cross entropy
    
    input:
        data_dict -- dictionary of binary data retrieved from data.txt
        train_step -- the model object
        cache -- for storing placeholder, weight and loss
        n_epoch -- number of epoch
        num4train -- number of training examples per epoch (count from beginning by order)
    
    output:
        train_step
    """
    # preset
    binary_dim = 8
    W = cache['W']
    b = cache['b']
    h = cache['h']
    h_= cache['h_']
    X = cache['X']
    Y = cache['Y']
    Y_ = cache['Y_']
    cell = cache['cell']
    loss = cache['loss']
    
    # load in a, b, c list
    a_list = data_dict['a']
    b_list = data_dict['b']
    c_list = data_dict['c']
    loss_list = []
    
    # start training
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
        
    for i in range(n_epoch):
        for j in range(num4train):
            if j % 10 == 0:
                print('Epoch: %d Example: %d is running...' % (i,j))
            a = np.array([a_list[j]], dtype=np.uint8)
            b = np.array([b_list[j]], dtype=np.uint8)
            c = np.array([c_list[j]], dtype=np.uint8)
            ab = np.c_[a,b]
            x = np.array(ab).reshape([1, binary_dim, 2])
            y = np.array(c).reshape([1, binary_dim])
            _, tmp_loss = sess.run([train_step, loss] , feed_dict = {X: x, Y: y})
            loss_list.append(tmp_loss)
            print('loss = {}'.format(tmp_loss))
            tmp_W = sess.run(W)
            print(tmp_W)
        
    # overwrite cache
    cache['W'] = W
    cache['b'] = b
    cache['h'] = h
    cache['h_'] = h_
    cache['X'] = X
    cache['Y'] = Y
    cache['Y_'] = Y_
    cache['cell'] = cell
    cache['loss'] = loss
    
    sess.close()
    return loss_list, train_step, cache

def pass_test_data(data_dict, cache, num4train, num4test, is_print = False):
    """
    Run test set on model. Open a session for running test set data
    Print out prediction, prob, loss for all test case
    Test set being [num4train+1: num4train + num4test]
    Define loss in one example = abs(y[i] - probs[i]) for i in [0:7]
    
    input:
        data_dict -- dictionary of binary data a_list, b_list
        cache -- for storing weight, placeholder, loss 
        sess -- tf.Session()
        num4train -- training set number 
        num4test -- test set number
    
    output:
        result -- list of [prediction, y, probs]
    """
    binary_dim = 8 
    # from data dict
    a_list = data_dict['a']
    b_list = data_dict['b']
    c_list = data_dict['c']
    
    # from cache
    loss = cache['loss']
    X = cache['X']
    Y = cache['Y']
    h = cache['h']
    remain_result = []
    
     # start training
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    for i in range(num4train + 1, num4train + num4test):
        a = a_list[i]
        b = b_list[i]
        c = c_list[i]
        ab = np.c_[a,b]
        x = np.array(ab).reshape([1, binary_dim, 2])
        y = np.array(c).reshape([1, binary_dim])
    
        # get predicted value
        [_probs, _loss] = sess.run([h, loss], feed_dict = {X: x, Y: y})
        probs = np.array(_probs).reshape([8])
        #print('probs = {}'.format(probs))
        prediction = np.array([1 if p >= 0.5 else 0 for p in probs]).reshape([8])
        # Save the result
        remain_result.append([prediction, y[0], probs])
    
        # calculate error
        error = np.sum(np.absolute(y - probs))
        
        if is_print:
            #print the prediction, the right y and the error.
            print('---------------')
            print('data index: {}'.format(i))
            print('probs:')
            print(probs)
            print('prediction:')
            print(prediction)
            print('y:')
            print(y[0])
            print('error')
            print(error)
            print('---------------')
            print()
    
    sess.close()
    return remain_result

def get_test_accuracy(remain_result):
    """
    check accuracy for test result. partial credit is given for wrong prediction
    * accuracy in one examples = sum([prediction[i] == y[i] for i in [0:7]) / 8
    
    input:
        remain_result -- np array of remaining result
    
    output:
        accuracy -- float of accuracy
    """
    accuracy = 0
    for i in range(len(remain_result)):
        len_ = len(remain_result[i][0])
        tmp_num = 0
        for j in range(len_):
            if remain_result[i][0][j] == remain_result[i][1][j]:
                tmp_num += 1
        accuracy += tmp_num / len_
    
    accuracy /= len(remain_result)
    
    print("Accuracy: %.4f"%(accuracy))
    return accuracy


## Testing
data_dict = txt_2_data(FILENAME)
train_step, cache = BasicRNN_Setup(4, 0.001)
train_step, cache = train_model(data_dict, train_step, cache, n_epoch = 3, num4train = 10)
remain_result = pass_test_data(data_dict, cache, 500, 10)
acc = get_test_accuracy(remain_result)