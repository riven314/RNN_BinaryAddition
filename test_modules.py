#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:09:33 2019

@author: alex Lau
"""
from data_util import *
from model_util import *
from metrics_util import *

binary_dim = 8    
batch_size = 64
n_epoch = 10
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
