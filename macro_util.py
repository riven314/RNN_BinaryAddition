#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:03:24 2019

Functions: wrap up data_util, model_util, metrics_util for macro training (input hyper-parameters)

Hyperparameter Description:
        filename -- str, data path
        batch_size -- int, minibatch size
        n_mb -- int, number of minibatch training for each epoch
        is_flip -- boolean, flip x, y upside down
        n_epoch -- int, number of epoch for training
        num4test -- int, number of test data. It is constrained to start index at 4000
        rnn_type -- str, type of rnn model, e.g basic_rnn, gru
        hidden_dims -- list, list of hidden dimensions. 
                       len is number of hidden layers, hidden_dims[i] is number of hidden neurons at layer i
        hidden_activation -- str, activation function for hidden layer, e.g tanh/ sigmoid/ relu/ elu/ leaky_relu
        init_method -- str, weight init method, e.g naive, he, xavier
        lr -- float, learning rate
        opt_method -- str, name of optimiser, e.g adam/ rmsprop/ nesterov_momentum/ momentum
        is_print_metrics --boolean,  whether to print metrics over iterations
        is_plot_metrics -- boolean, whether to plot loss curve and accuracy curve at the end with 
        suff_seed -- int, better >= 100. Perform shuffling with random seed = shuff_seed right after data load in

@author: Alex Lau
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_util import *
from model_util import *
from metrics_util import *

def macro_run(input_cache):
    """
    wrap_train() caller. Input is wrapped into dictionary form. 
    The function display training result. 
    
    input:
        input_cache -- dict, hyper-parameters dictionary. input similar to one in wrap_train. key in str
        
    output:
        None
    """
    filename = input_cache['filename']
    batch_size = input_cache['batch_size']
    n_mb = input_cache['n_mb']
    num4test = input_cache['num4test']
    n_epoch = input_cache['n_epoch']
    is_flip = input_cache['is_flip']
    rnn_type = input_cache['rnn_type']
    hidden_dims = input_cache['hidden_dims']
    hidden_activation = input_cache['hidden_activation']
    init_method = input_cache['init_method']
    lr = input_cache['lr']
    opt_method = input_cache['opt_method']
    is_print_metrics = input_cache['is_print_metrics']
    is_plot_metrics = input_cache['is_plot_metrics']
    shuff_seed = input_cache['shuff_seed']
    
    test_acc, test_cost = wrap_train(filename = filename, batch_size = batch_size, n_mb = n_mb, n_epoch = n_epoch, is_flip = is_flip, num4test = num4test,
                                                   rnn_type = rnn_type, hidden_dims = hidden_dims, hidden_activation = hidden_activation, init_method = init_method,
                                                   lr = lr, opt_method = opt_method, is_print_metrics = is_print_metrics, is_plot_metrics = is_plot_metrics,
                                                   shuff_seed = shuff_seed)
    return test_acc, test_cost

def wrap_train(filename, batch_size, n_mb, n_epoch, is_flip, num4test,
               rnn_type, hidden_dims, hidden_activation, init_method, 
               lr, opt_method, is_print_metrics = False, is_plot_metrics = True,
               shuff_seed = None):
    """
    wrap up data_util, model_util, metrics_util. The function is controlled by hyper-parameters
    num4train = n_mb * batch_size, it is constrained to start index at 0
    FILENAME default to be 'data/data.txt'
    
    input:
        filename -- str, data path
        batch_size -- int, minibatch size
        n_mb -- int, number of minibatch training for each epoch
        is_flip -- boolean, flip x, y upside down
        n_epoch -- int, number of epoch for training
        num4test -- int, number of test data. It is constrained to start index at 4000
        rnn_type -- str, type of rnn model, e.g basic_rnn, gru
        hidden_dims -- list, list of hidden dimensions. 
                       len is number of hidden layers, hidden_dims[i] is number of hidden neurons at layer i
        hidden_activation -- str, activation function for hidden layer, e.g tanh/ sigmoid/ relu/ elu/ leaky_relu
        init_method -- str, weight init method, e.g naive, he, xavier
        lr -- float, learning rate
        opt_method -- str, name of optimiser, e.g adam/ rmsprop/ nesterov_momentum/ momentum
        is_print_metrics --boolean,  whether to print metrics over iterations
        is_plot_metrics -- boolean, whether to plot loss curve and accuracy curve at the end with 
        suff_seed -- int, better >= 100. Perform shuffling with random seed = shuff_seed right after data load in
        
    output:
        last_test_acc -- float, accuracy for test set at the end of training
        last_test_cost -- float, cost for test set at the end of training
    """
    # Set up other hyper-parameters
    num4train = batch_size * n_mb
    
    # Set up training 
    tf.reset_default_graph()
    train_X, train_Y = init_placeholder()
    parameters = init_parameters(hidden_dim = 4, init_method = init_method, rand_seed = 1)
    
    # Select RNN model type
    if rnn_type == 'basic_rnn':
        rnn_cell = create_BasicRNN(hidden_dim = hidden_dims, activation = hidden_activation)
    elif rnn_type == 'gru':
        rnn_cell = create_GRU(hidden_dim = hidden_dims, activation = hidden_activation)
    else:
        print('input valid rnn_type e.g basic_rnn/ gru')
        return None
    # Forward propagation
    outputs, state = forward_timestep(train_X, rnn_cell, batch_size = batch_size)
    train_h = forward_output(train_X, outputs, parameters)
    train_cost = compute_cost(train_h, train_Y)
    train_step = backpropagate_optimise(train_cost, lr = lr, optimiser = opt_method)

    # Set up test set
    test_X, test_Y = init_placeholder()
    test_output, _ = forward_timestep(test_X, rnn_cell, batch_size = num4test)
    test_h = forward_output(test_X, test_output, parameters)
    test_cost = compute_cost(test_h, test_Y)

    # Load in data
    data_dict = txt_2_dict(filename)
    x, y = dict_2_nparr(data_dict, is_flip = is_flip)
    
    # perform shuffling after data loading, it is used to change test set
    if shuff_seed is not None:
        x, y = shuffle(x, y, rand_seed = shuff_seed)
    
    # partitioning training set and test set
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
        if is_print_metrics:
            print('Epoch: %d is running...' % i)
        for j in range(n_mb):
            mb_x, mb_y = get_minibatches(train_x, train_y, j, batch_size)
            # training set
            _, train_bs_probs, train_bs_cost = sess.run([train_step, train_h, train_cost], 
                                                        feed_dict = {train_X: mb_x, train_Y: mb_y})
            train_cost_list.append(train_bs_cost)
            train_preds = get_prediction(train_bs_probs)
            train_acc = get_accuracy(mb_y, train_preds)
            train_acc_list.append(train_acc)
            # test set
            test_all_probs, test_all_cost = sess.run([test_h, test_cost], 
                                                     feed_dict = {test_X: test_x, test_Y: test_y})
            test_cost_list.append(test_all_cost)
            test_all_preds = get_prediction(test_all_probs)
            test_acc = get_accuracy(test_y, test_all_preds)
            test_acc_list.append(test_acc)
            # print out metrics for train and test set
            if is_print_metrics:
                print('train cost: %.4f, test cost: %.4f | train acc: %.4f, test acc: %.4f' % (train_bs_cost, test_all_cost, train_acc, test_acc))
    sess.close()
    
    # print out metrics and plot
    last_test_acc, last_test_cost = test_acc_list[-1], test_cost_list[-1]
    if is_plot_metrics:
        get_metrics_plot(train_cost_list, test_cost_list, train_acc_list, test_acc_list)
        print_metrics(last_test_acc, last_test_cost)
    return last_test_acc, last_test_cost

if __name__ == '__main__':
    # For testing
    hyperparam_cache = {'filename': 'data/data.txt',
                        'batch_size': 64,
                        'n_mb': 10,
                        'num4test': 1000,
                        'n_epoch': 10,
                        'is_flip': True,
                        'rnn_type': 'gru',
                        'hidden_dims': [4],
                        'hidden_activation': 'tanh',
                        'init_method': 'xavier',
                        'lr': 0.01, 
                        'opt_method': 'adam',
                        'is_print_metrics': True,
                        'is_plot_metrics': True,
                        'shuff_seed': None}
    macro_run(hyperparam_cache)