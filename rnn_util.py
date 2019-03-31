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

# a, b, c * 5000
FILENAME = 'data.txt'

def txt_2_dictaa(filename):
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
    
    method
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

def predict(X, outputs, parameters):
    """
    find out accuracy rate for test data X 
    feed X into propagation through timestep, then it goes to forward feed to get final prediction
    
    input
    """    
    x = tf.placeholder('float', X.get_shape().as_list())
    z = forward_pass(x, outputs, parameters)
    
    sess = tf.Session()
    prediction = sess.run(z, feed_dict = {x: X})
    sess.close()
    return prediction


#####################################################
#####################################################    
### TESTING
batch_size = 16
tf.reset_default_graph()
X, Y = init_placeholder()
parameters = init_parameters(hidden_dim = 4)
rnn_cell = create_BasicRNN(hidden_dim = 4)
outputs, state = forward_timestep(X, rnn_cell, batch_size = batch_size)
h = forward_output(X, outputs, parameters)
cost = compute_cost(h, Y)
train_step = backpropagate_optimise(cost)

# pull data
data_dict = txt_2_dict(FILENAME)
a_list = data_dict['a']
b_list = data_dict['b']
c_list = data_dict['c']

# start training
binary_dim= 8
n_epoch = 2
num4train = 3
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#####################################################
#####################################################  
# GET TEST DATA AND DEFINE PROPOGATION GRAPH FOR GETTING LOSS
a_test = np.array(a_list[4000:4100], dtype=np.uint8)
b_test = np.array(b_list[4000:4100], dtype=np.uint8)
c_test = np.array(c_list[4000:4100], dtype=np.uint8)
ab_test = np.c_[a_test,b_test]
# x = (batch_size, time_steps, input_dim)
x_test = np.array(ab_test).reshape([100, 2, binary_dim])
x_test = np.transpose(x_test, (0, 2, 1))
# y = (batch_size, time_steps)
y_test = np.array(c_test).reshape([100, binary_dim])
X_test, Y_test = init_placeholder()
test_output, _ = forward_timestep(X_test, rnn_cell, batch_size = 100)
h_test = forward_output(X_test, test_output, parameters)
test_cost = compute_cost(h_test, Y_test)

cost_list = []
for i in range(n_epoch):
    for j in range(num4train):
        if j % 10 == 0:
            print('Epoch: %d Example: %d is running...' % (i,j))
        a = np.array(a_list[:batch_size], dtype=np.uint8)
        b = np.array(b_list[:batch_size], dtype=np.uint8)
        c = np.array(c_list[:batch_size], dtype=np.uint8)
        ab = np.c_[a,b]

        # x = (batch_size, time_steps, input_dim)
        x = np.array(ab).reshape([batch_size, 2, binary_dim])
        x = np.transpose(x, (0, 2, 1))
        
        # y = (batch_size, time_steps)
        y = np.array(c).reshape([batch_size, binary_dim])
        _, tmp_cost = sess.run([train_step, cost] , feed_dict = {X: x, Y: y})
        cost_list.append(tmp_cost)
        print('train_cost = {}'.format(tmp_cost))
        W = parameters['W']
        tmp_W = sess.run(W)
        print(tmp_W)
        
        # get cost from test set
        tmp_cost1, tmp_cost2 = sess.run([test_cost, test_cost] , feed_dict = {X_test: x_test, Y_test: y_test})
        print('test_cost1 = {}'.format(tmp_cost1))
        print('test_cost2 = {}'.format(tmp_cost2))



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
    # tf.nn.dynamic_rnn propagate RNNCell by time-step
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
            _, tmp_loss = sess.run([loss] , feed_dict = {X: x, Y: y})
            compute_cost
        
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
data_dict = txt_2_dict(FILENAME)
train_step, cache = BasicRNN_Setup(4, 0.001)
train_step, cache = train_model(data_dict, train_step, cache, n_epoch = 3, num4train = 10)
remain_result = pass_test_data(data_dict, cache, 500, 10)
acc = get_test_accuracy(remain_result)
