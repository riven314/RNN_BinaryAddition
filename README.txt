NAME: LAU WAN HONG
UID: 3035098070
CURRICULUM: MDASC(FT)

REQUIRED
python (3.6.7)
tensorflow (1.5.0)
numpy
matplotlib

OVERVIEW
data_util.py: utility functions for handling data transformation (details documented in data_util.py)
model_util.py: utility functions for handling tf model setup (details documented in model_util.py)
metrics_util.py: utility functions for handling metrics reporting
macro_util.py: consolidate the above utility functions into a macro function
hyperparameters_analysis.ipynb: test performance with different hyper-parameters by macro_util.py
finalmodel_robust.ipynb: use the final model found in hyperparameters_analysis.ipynb for full run and test for robustness

CHOICE OF HYPERPARAMETER
- minibatch with shuffling is used 
- learning rate = 0.01
- optimiser = adam
- basic RNN with hidden layers [4, 4, 4]
- tanh activation function is used in hidden layers
- data x, y are flipped upside down (e.g [0,1,0,0,0,0,0,0] >> [0,0,0,0,0,0,1,0]) 
** datailed explanation on the choice of hyper-parameters are documented on hyperparameters_analysis.ipynb

TO RUN THE SCRIPT WITH OTHER TEST CASES
Use model_util.py for main run. See hyperparameters_analysis.ipynb, finalmodel_robust.ipynb for example of how to use it
Please edit the data path in wrap_train() from macro_util.py

REFERENCE
I make reference to the following materials when writing the code
1. Deep Learning Specialisation, Andrew Ng, Coursera
2. Zhihu introduction to tensorflow implementation for RNN 
    https://zhuanlan.zhihu.com/p/44424550
3. step by step instruction for tf implementation of RNN 
    https://www.guru99.com/rnn-tutorial.html
3. many-to-many RNN implementation by tensorflow github 
    https://github.com/easy-tensorflow/easy-tensorflow/blob/master/7_Recurrent_Neural_Network/Tutorials/05_Many_to_Many.ipynb
4. how to define weight in dynamic_run() 
    https://stackoverflow.com/questions/43696892/how-to-set-different-weight-variables-in-multiple-rnn-layers-in-tensorflow
5. github source code with demo on single layer RNN and multilayer RNN (500-527) 
    https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/rnn.py 
6. different weight init method with tensorflow implementation and math proof
    https://medium.com/@prateekvishnu/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528
7. RNN-LSTM setup in tensorflow(dropout, batch norm, multicell)
    http://lawlite.me/2017/06/21/RNN-LSTM%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C-03Tensorflow%E8%BF%9B%E9%98%B6%E5%AE%9E%E7%8E%B0/
8. Medium: working with multilayer BasicRNNCell
    https://medium.com/@d.vitonyte/dynamic-multi-cell-rnn-with-tensorflow-b3627bcd2f47



