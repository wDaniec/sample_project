# sample_project


A simple project inspired by gmum/toolkit for pytorch lightning. The code reproduces the effect known as warm-starting.
The network pretrained on half of the dataset generalises slightly worse than a network trained on the whole dataset from scratch.
Even though both achieve a training accuracy of 100%. It doesn't hold with all architectures. The effect is not achieved for example
with simple, linear regression. Though it's hard to categorise such architecture as a machine learning model.

All avaiable models that can be used for this experiment are kept in src/models folder.
The cifar dataset is inn the src/data folder. The code is very easy to use and is divided into meta categories like
datasets/models/modules.

To use neptue, the env_variables have to be set in e.sh.
To start training use: source e.sh and then python main.py. The libraries have to be installed manually.
