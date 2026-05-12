
## Tensorflow versions and `tf_gnns`

|tf   | Tests status |
|-----|----|
|2.13 | ![alt-img](https://raw.githubusercontent.com/mylonasc/tf_gnns/refs/heads/main/doc/shields/tf2.13.svg) | 
|2.14 | ![alt-img](https://raw.githubusercontent.com/mylonasc/tf_gnns/refs/heads/main/doc/shields/tf2.14.svg) | 
|2.15 | ![alt-img](https://raw.githubusercontent.com/mylonasc/tf_gnns/refs/heads/main/doc/shields/tf2.15.svg) | 
|2.17 | ![alt-img](https://raw.githubusercontent.com/mylonasc/tf_gnns/refs/heads/main/doc/shields/tf2.17.svg) | 

<details>
<summary>Further info</summary>
There are some unresolved bugs with the latest TensorFlow versions. Due to the ongoing transition from Keras 2 to Keras 3, some problems are already resolved, but there are still flaky parts in the code.
Since I develop this library single-handedly, I will wait for the dust to settle with Keras 3.
At the moment, it is recommended to use `tensorflow==2.15` or earlier.
</details>

# `tf_gnns` - A Hackable GraphNets library
![alt-img](https://raw.githubusercontent.com/mylonasc/tf_gnns/main/doc/figures/tfgnns_logo2.png)
A library for easy construction of message-passing networks in TensorFlow Keras.
It is largely inspired by this [DeepMind paper](https://arxiv.org/abs/1806.01261) and the corresponding open-source library ([original graph_nets library](https://github.com/deepmind/graph_nets)).

The `tf_gnns` library has no external dependencies except TensorFlow 2.x (there is no support for TF 1.x graph/session-based computation).
It implements alternative design constraints from `graph_nets`, taking advantage of Keras facilities to build complex models easily and without large drops in performance.

`tf_gnns` is built to support arbitrary node/edge/global attributes and update functions.
A set of utility functions for MLP construction with Keras is also provided (i.e., handling input/output sizes for valid networks), replacing Sonnet.

The main motivation for this library was the absence of a relatively short and efficient implementation of GNNs explicitly created to take advantage of Keras functionality.
GNN implementations that take advantage of `tensorflow_probability` functionality are planned for future releases (such as the one used in [2]).

## Installing `tf_gnns`
---
**NOTE**

Currently `tensorflow==2.17` and `tensorflow_probability==0.24` have one failing test. The failure is related to validation between two different computation modes: `GraphTuple` (efficient) and `Graph` (less efficient; does not use `unsorted_segment_sum` and other sparse aggregations). I have not resolved the source of the issue yet, but it could be a benign deviation due to operation-order changes in low-level kernels.
All tests pass with `tensorflow==2.15` and `tensorflow_probability==0.22`, and it is therefore recommended to use these. 

---

Install with `uv` (recommended):
```
uv sync
```

Or install with `pip`:
```
# optional - recommended:
# pip install tensorflow==2.15 
# pip install tensorflow_probability==0.22
pip install tf_gnns
```
## Use through Docker

You can build a Docker image that uses `tf_gnns` with the following command, based on Ubuntu 22:

```
docker build . -t tf_gnns_215 --network host  --build-arg TENSORFLOW_VERSION=2.15
```

The container implements some logic to sort out the necessary dependencies. Namely, 
 * Numpy 1.x is required for tf <= 2.14
 * Keras 2 support needs to be enabled for tf >= 2.16
 * The `tensorflow_probability` version is selected through a mapping given the tensorflow version.


# Examples

## `tf_gnns` basics
You can inspect some basic functionality in the following Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mylonasc/tf_gnns/blob/main/notebooks/01_tf_gnn_basics.ipynb)


## List sorting example
(Example from the original `deepmind/graph_nets` library)
If you are familiar with the original `graph_nets` library, this example will help you understand how you can transition to `tf_gnns`.

Sort a list of elements.
This notebook and the accompanying code demonstrates how to use the Graph Nets library to learn to sort a list of elements.

A list of elements is treated as a fully connected graph between the elements. 
The network is trained to label the start node, and which (directed) edges correspond to the links to the next largest element, for each node.

After training, prediction ability is tested by comparing output to true sorted lists. Then the network's ability to generalize is tested by using it to sort larger lists.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mylonasc/tf_gnns/blob/main/notebooks/02_list_sorting.ipynb)

## Protein-Protein Interaction example
This example shows how to adapt `torch_geometric` (aka PyG) inputs to `tf_gnns` inputs.
The notebook can be run end-to-end in Google Colab, and out of the box it gives a test-set F1 score that is competitive with SOTA.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mylonasc/tf_gnns/blob/main/notebooks/03_ProteinProteinInteraction_MPNN.ipynb)

## Performance
From initial tests, the performance of `tf_gnns` seems to be at least as good as `deepmind/graph_nets` when using tensor dictionaries.

# Publications using `tf_gnns`
The library has been used so far in the following publications:

\[1\] [Bayesian graph neural networks for strain-based crack localization](https://arxiv.org/abs/2012.06791) 

\[2\] [Remaining Useful Life Estimation Under Uncertainty with Causal GraphNets](https://arxiv.org/abs/2011.11740)

\[3\] [Relational VAE: A Continuous Latent Variable Model for Graph Structured Data](https://arxiv.org/abs/2106.16049)
