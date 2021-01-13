# `tf_gnns` - A Hackable GraphNets library
A library for easy construction of message-passing networks in tensorflow keras.
It is inspired largely from [DeepMind paper](https://arxiv.org/abs/1806.01261) where a library was also open-sourced [original graph_nets library](https://github.com/deepmind/graph_nets)

The `tf_gnns` library has no external dependencies except tensorflow 2.x (there is no support for tf 1.x sessions-based computation). 
It implements some alternative design constraints from `graph_nets` taking advantage of some conviniences keras provides. 


In `tf_gnns` there are few restrictions on what may a node function, an edge function and an aggregation function. Moreover, very few sanity checks are 
performed so it's possibly easier to make a mistake if you do not know what you are doing exactly! Nevertheless, it is potentially easier to implement advanced 
features while (potentially) having better supervision on what is going on if you are familiar with the functional API of `tf.keras` (hence *hackable*).

# Examples
More examples to be implemented. The library has been used so far in the following publications:

You may inspect some basic functionality on the following colab notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mylonasc/tf_gnns/blob/main/notebooks/01_tf_gnn_basics.ipynb)

\[1\][Bayesian graph neural networks for strain-based crack localization](https://arxiv.org/abs/2012.06791) 

\[2\][Remaining Useful Life Estimation Under Uncertainty with Causal GraphNets](https://arxiv.org/abs/2011.11740)




