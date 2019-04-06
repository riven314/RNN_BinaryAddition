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
9. diminishing learning rate
10. add regularization
11. input data upside down
12. ** test with small data to see if it will get overfitting without regularization

Issue:
1. how to keep track fo the weight for BasicRNNCell
2. x = np.array(ab).reshape([1, binary_dim, 2]) << this is problematic 
3. take the data from top to bottom or from bottom to top?
4. at last batch may not fit the specified batch size, how to do?

Further:
1. use tensorboard to keep track of key metrics
2. Plot loss function

[27/03/2019]
TO DO LIST
1. compute function for compute loss function
2. function for test case evaluation
3. function for initialising weight
4. function for creating placeholder
5. function for forward propagation
6. function for shuffling
7. optimiser function
8. RNN propagation via time function
ACTUAL:
1. compute loss function
2. initial weight function
3. create placeholder function
4. forward propagation
5. optimiser function
6. RNN propagation
7. training loop

[28/03/2019]
ACTUAL:
1. shuffle function
2. transform np array function

[31/03/2019]
1. Working on prediction and accuracy function

[05/04/2019]
1. Complete get_minibatches() function
2. Complete refactor the training loop
3. Complete plotting function
FOLLOW UP
1. Set up dropout
2. Set up multi layer
3. upside down read in data
4. modularisation

[06/04/2019]
1. Modularized the functions (data_util, model_util, metrics_util)


Point to Note:
1. tf version = 1.5.0

Reference
- Some utility fucntinos are adopted from Andrew Ng Deep Learning Specialisation
- Zhihu introduction to tensorflow implementation for RNN 
    https://zhuanlan.zhihu.com/p/44424550
- step by step instruction for tf implementation of RNN 
    https://www.guru99.com/rnn-tutorial.html
- many-to-many RNN implementation by tensorflow github 
    https://github.com/easy-tensorflow/easy-tensorflow/blob/master/7_Recurrent_Neural_Network/Tutorials/05_Many_to_Many.ipynb
- how to define weight in dynamic_run() 
    https://stackoverflow.com/questions/43696892/how-to-set-different-weight-variables-in-multiple-rnn-layers-in-tensorflow
- github source code with demo on single layer RNN and multilayer RNN (500-527) 
    https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/rnn.py 
- set dynamic_rnn without fixed batch size 
    https://stackoverflow.com/questions/45048511/how-to-set-tensorflow-dynamic-rnn-zero-state-without-a-fixed-batch-size
- tensorflow doc on RNN iterataion training
    https://www.tensorflow.org/tutorials/sequences/recurrent
- difference between dynamic_rnn and rnn 
    https://www.zhihu.com/question/52200883
- different weight init method with tensorflow implementation and math proof
    https://medium.com/@prateekvishnu/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528

"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

def dict_2_nparr(data_dict):
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


def init_parameters(hidden_dim, output_dim = 1, init_method = 'naive', rand_seed = 1):
    """
    initialised weights and bias, then store in dictionary i.e parameters.
    it creates the weight and bias for the last layer (right before the output)
    the weights and biases in intermediate layers are defined in RNN object construction by default
    
    input:
        hidden_dims -- dimension of the last hidden layer
        output_dim -- dimension of the output
        init_method -- method for initialisation. 
                        naive = original method
                        xavier = np.random.randn() * np.sqrt(1./layers_dim[l-1]) ~~ generally used for tanh
                        he = np.random.randn() * np.sqrt(2./layers_dim[l-1]) ~~ generally used for relu
                        
    output:
        parameters -- dictionary storing weight matrix and bias vector for all layers
    """
    parameters = {}
    # Stream-line different init method
    if init_method == 'naive':
        W = tf.Variable(tf.random_uniform([hidden_dim, output_dim], minval=- 1.0, maxval = 1.0), name = 'W')
        b = tf.Variable(tf.zeros([1, output_dim]), name='b')
    elif init_method == 'xavier':
        W = tf.get_variable('W', 
                            [hidden_dim, output_dim], 
                            initializer = tf.contrib.layers.xavier_initializer(seed = rand_seed))
        b = tf.get_variable('b', 
                            [1, output_dim], 
                            initializer = tf.contrib.layers.xavier_initializer(seed = rand_seed))
    elif init_method == 'he':
        W = tf.get_variable('W',
                            [hidden_dim, output_dim],
                            initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32))
        b = tf.get_variable('b',
                            [1, output_dim],
                            initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32))
    else:
        print('Invalid init_method input!')
        return None
    # store weights and bias in parameters
    parameters['W'] = W
    parameters['b'] = b
    return parameters

def init_placeholder(input_dim = 2, time_steps = 8):
    """
    create placeholder for storing input data and label data
    
    input:
        input_dim -- dimension of input i.e by default 1 from a and 1 from b = 2 in total
        time_steps -- time steps i.e by default 8
        
    output:
        X -- input data, tensor of shape (batch size, time steps, input dim)
        Y -- label data, tensor of shape (batch, time step)
    """
    X = tf.placeholder(tf.float32, [None, time_steps, input_dim], name='x')
    Y = tf.placeholder(tf.float32, [None, time_steps], name='y')
    return X, Y

def create_BasicRNN(hidden_dim, activation = 'sigmoid'):
    """
    create BasicRNNCell object with specified activation function
    
    input:
        hidden_dim -- dimension of the last hidden layer
        activation -- activation function used in the hidden layers i.e sigmoid/ relu/ elu/ leaky_relu/ tanh
        
    output:
        rnn_cell -- RNN cell object
    """
    # create basic RNN object with specified activation for hidden layer
    if activation == 'sigmoid':
        rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim, activation= tf.nn.sigmoid)
    elif activation == 'relu':
        rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim, activation= tf.nn.relu)
    elif activation == 'leaky_relu':
        rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim, activation= tf.nn.leaky_relu)
    elif activation == 'elu':
        rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim, activation= tf.nn.elu)
    elif activation == 'tanh':
        rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim, activation= tf.nn.tanh)
    return rnn_cell    
    

def forward_timestep(X, rnn_cell, time_steps = 8, batch_size = 1):
    """
    initial state is defined and the RNN network is propagated to the last layer
    weight and bias in hidden layers are calculated
    How those weights and biases are initialised are unknown
    
    input:
        X -- input data, tensor of shape (batch size, time steps, input dim)
        rnn_cell -- RNN model object 
        time_steps -- time steps for each examples
        batch_size -- batch size of the examples
    output
        outputs -- final output from the network for prediction processing, it is then computed with W, b and activation for prediction
                    it is a tensor with shape (batch size, time steps, hidden dim)
        state -- the state generated from the last time step
    """
    # set up initial state
    initial_state = rnn_cell.zero_state(batch_size, dtype = tf.float32)

    # propagate RNNCell by time-step and by default create weight in hidden layers    
    # output is a tensor of shape [batch_size, time_steps, hidden_dim]
    # state is a tensor of shape [batch_size, hidden_dim]
    outputs, state = tf.nn.dynamic_rnn(cell = rnn_cell, 
                                       inputs = X, 
                                       dtype = tf.float32,
                                       initial_state = initial_state)
    return outputs, state

def forward_output(X, outputs, parameters):
    """
    do forward propagation to get output of the network
    the step is done after propagate_timesteps()
    h = g(outputs * W1 + b), where g(.) is sigmoid function
    
    input:
        X -- input data, tensor of shape (batch size, time steps, input dim)
        outputs -- output from dynamic_rnn object, a tensor with shape (batch size, time steps, hidden dim)
        parameters -- dictionary of parameters i.e W and b at last layer
    output:
        h -- output of forward propagation
    """
    #_, time_steps, hidden_dim = outputs.get_shape().as_list()
    #output_reshape = tf.reshape(outputs,[time_steps, hidden_dim])

    W = parameters['W']
    b = parameters['b']
    
    # tf.tensordot: 
    # (batch size, time_steps, hidden_dim) * (hidden_dim, output_dim) = (batch size, time_steps, output_dim)
    h = tf.nn.sigmoid(tf.tensordot(outputs, W, axes=[[2], [0]]) + b, name='h')
    return h

def compute_cost(h, Y):
    """
    compute the cost
    the loss for an example = sum of binary cross entropy for each timestep in an example
    the cost for the examples = average binary cross entropy over batch examples
    
    input:
        h -- output of forward propagation
    output:
        Y -- label data, tensor of shape (batch, time step)
    """
    # reshape h and Y tensor with shape (batch_size, time_steps)
    batch_size, time_steps, _ = h.get_shape().as_list()
    h_ = tf.reshape(h, [batch_size, time_steps])
    Y_ = tf.reshape(Y, [batch_size, time_steps])
    #h_ = tf.reshape(h, [time_steps])
    #Y_ = tf.reshape(Y, [time_steps])
    
    # sum across columns >> (batch_size, )
    cost = tf.reduce_sum(-Y_ * tf.log(h_) - (1-Y_) * tf.log(1-h_), name='loss', axis = 1)
    # average by batch size
    cost = tf.reduce_mean(cost)
    return cost

def backpropagate_optimise(cost, lr = 0.01, optimiser = 'adam'):
    """
    backpropagation with specified optimisation method
    
    input:
        cost -- cost function from compute_cost()
        optimiser -- string indicating optimiser e.g adam/ rmsprop/ nesterov_momentum/ momentum
    
    output:
        train_step -- optimisation steps
    """
    # DEFINE OPTIMISER AND LEARNING RATE
    if optimiser == 'adam':
        train_step = tf.train.AdamOptimizer(lr).minimize(cost)
    elif optimiser == 'rmsprop':
        train_step = tf.train.RMSPropOptimizer(lr).minimize(cost)
    elif optimiser == 'nesterov_momentum':
        train_step = tf.train.MomentumOptimizer(learning_rate = lr, use_nesterov = True).minimise(cost)
    elif optimiser == 'momentum':
        train_step = tf.train.MomentumOptimizer(learning_rate = lr, use_nesterov = False).minimise(cost)
    return train_step

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
    
        
#####################################################
#####################################################  
binary_dim = 8    
batch_size = 64
n_epoch = 300
n_mb = 20
num4train = batch_size * n_mb
num4test = 100

### TRAINING SESSION
tf.reset_default_graph()
train_X, train_Y = init_placeholder()
parameters = init_parameters(hidden_dim = 4)
rnn_cell = create_BasicRNN(hidden_dim = 4)
outputs, state = forward_timestep(train_X, rnn_cell, batch_size = batch_size)
train_h = forward_output(train_X, outputs, parameters)
train_cost = compute_cost(train_h, train_Y)
train_step = backpropagate_optimise(train_cost)

### TESTING SESSION
test_X, test_Y = init_placeholder()
test_output, _ = forward_timestep(test_X, rnn_cell, batch_size = num4test)
test_h = forward_output(test_X, test_output, parameters)
test_cost = compute_cost(test_h, test_Y)

# pull data
data_dict = txt_2_dict(FILENAME)
x, y = dict_2_nparr(data_dict)
train_x, train_y, test_x, test_y = partition_data(x, y, 
                                                  train_idx = 0, num4train = num4train, 
                                                  test_idx = 4000, num4test = num4test)

# start training
sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_cost_list, test_cost_list = [], []
train_acc_list, test_acc_list = [], []
for i in range(n_epoch):
    train_x, train_y = shuffle(train_x, train_y, rand_seed = i)
    for j in range(n_mb):
        mb_x, mb_y = get_minibatches(train_x, train_y, j, batch_size)
        if j % 10 == 0:
            print('Epoch: %d Example: %d is running...' % (i,j))
        # training set
        _, train_bs_probs, train_bs_cost = sess.run([train_step, train_h, train_cost], 
                                                    feed_dict = {train_X: mb_x, train_Y: mb_y})
        train_cost_list.append(train_bs_cost)
        train_preds = get_prediction(train_bs_probs)
        train_acc = get_accuracy(mb_y, train_preds)
        train_acc_list.append(train_acc)
        print('train_cost = {} | train_acc = {}'.format(train_bs_cost, train_acc))
        # test set
        test_all_probs, test_all_cost = sess.run([test_h, test_cost] , feed_dict = {test_X: test_x, test_Y: test_y})
        test_cost_list.append(test_all_cost)
        test_all_preds = get_prediction(test_all_probs)
        test_acc = get_accuracy(test_y, test_all_preds)
        test_acc_list.append(test_acc)
        print('test_cost = {} | test_acc = {}'.format(test_all_cost, test_acc))
