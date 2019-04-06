#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 11:57:49 2019

Functions:
    1. init weights and placeholders (diff weight init methods)
    2. forward propagation for the network
    3. backward propagatino for the network (diff optimisation method)
    4. create RNN blocks and do timestep feeding (diff RNN types: Basic RNN)
    5. compute cost for the network

@author: Alex Lau
"""
import tensorflow as tf

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
