# `tf_gnns` - A Hackable GraphNets library
![alt-img](https://raw.githubusercontent.com/mylonasc/tf_gnns/main/doc/figures/tfgnns_logo2.png)
A library for easy construction of message-passing networks in tensorflow keras.
It is inspired largely by this [DeepMind paper](https://arxiv.org/abs/1806.01261) and the corresponding open-sourced library ([original graph_nets library](https://github.com/deepmind/graph_nets))

The `tf_gnns` library has no external dependencies except tensorflow 2.x (there is no support for tf 1.x graphs/sessions-based computation). 
It implements some alternative design constraints from `graph_nets` taking advantage of some facilities keras provides to make complex models easily and without large drops in performance.

In `tf_gnns` there are no restrictions on what may a node function, an edge function and an aggregation function be.
Moreover, few sanity checks are performed, so it is possibly easier to make a mistake if you do not know what you are doing exactly. 
Nevertheless, it is easier to implement advanced features and having better supervision on what is going on if you are familiar
 with the functional API of `tf.keras` (hence *hackable*). 

The main motivation for this library was the absense of a short implementation of GNNs that was explicitly created to take advantage of keras's functionalities.
GNN implementations which take advantage of `tensorflow_probability` functionality is to be released in the future (as the one used in \[2\]).

## Installing `tf_gnns`
Install with `pip`:
```
pip install tf_gnns
```

# Examples
More examples to be implemented. Feel free to contribute!

You may inspect some basic functionality on the following colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mylonasc/tf_gnns/blob/main/notebooks/01_tf_gnn_basics.ipynb)

# Publications using `tf_gnns`
The library has been used so far in the following publications:

\[1\][Bayesian graph neural networks for strain-based crack localization](https://arxiv.org/abs/2012.06791) 

\[2\][Remaining Useful Life Estimation Under Uncertainty with Causal GraphNets](https://arxiv.org/abs/2011.11740)




