NAME: LAU WAN HONG
UID: 3035098070
CURRICULUM: MDASC(FT)

REQUIRED
python (3.6.7)
tensorflow (1.5.0)
numpy
matplotlib

OVERVIEW:
data_util.py: utility functions for handling data transformation (details documented in data_util.py)
model_util.py: utility functions for handling tf model setup (details documented in model_util.py)
metrics_util.py: utility functions for handling metrics reporting
macro_util.py: consolidate the above utility functions into a macro function
hyperparameters_analysis.ipynb: test performance with different hyper-parameters by macro_util.py
finalmodel_robust.ipynb: use the final model found in hyperparameters_analysis.ipynb for full run and test for robustness

CHOICE OF HYPERPARAMETER:
- minibatch with shuffling is used 
- learning rate = 0.01
- optimiser = adam
- basic RNN with hidden layers [4, 4, 4]
- tanh activation function is used in hidden layers
- data x, y are flipped upside down (e.g [0,1,0,0,0,0,0,0] >> [0,0,0,0,0,0,1,0]) 
** datailed explanation on the choice of hyper-parameters are documented on hyperparameters_analysis.ipynb

TO RUN THE SCRIPT WITH OTHER TEST CASES:
Use model_util.py for main run. See hyperparameters_analysis.ipynb, finalmodel_robust.ipynb for example of how to use it
Please edit the data path in wrap_train() from macro_util.py

REFERENCE:
I make reference to the following materials when writing the code
1. 

