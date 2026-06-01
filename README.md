# `tf_gnns` - A Hackable GraphNets library
![alt-img](https://raw.githubusercontent.com/mylonasc/tf_gnns/main/docs/figures/tfgnns_logo2.png)
A library for easy construction of message-passing networks in Keras 3.

It is largely inspired by this [DeepMind paper](https://arxiv.org/abs/1806.01261) and the corresponding open-source library ([original graph_nets library](https://github.com/deepmind/graph_nets)).
In addition it contains baseline tested implementations for GCNs. 

The `tf_gnns` library has no external dependencies except Keras 3 and a deep learning backend (tf, torch, and jax supported and tested). 

### Initial motivation
This library was initially implemented for GraphNet-style MPNNs and all the other related architectures that can be seen as special cases of Graphnets. 

It has slighly different design constraints from the original DeepMind `graph_nets`, since it is taking advantage of Keras facilities to build complex models easily and without large drops in performance.

`tf_gnns` is built to support arbitrary node/edge/global attributes and update functions. 

### Philosophy and performance

This `tf_gnns` framework explicitly avoids containing custom low-level kernels, as framework operations are often more maintainable and well performing.

The `tf_gnns` computations are formed in a way that allow the graph compilers to create optimized GPU code, that can be reasonably expected to perform as good as the underlying keras-compiled code allows.
Often this is close to the limits of the capabilities of the accelerators. 

The framework takes advantage of graph computation where available (e.g., jax and tensorflow mainly) for creating fused networks. In preliminary benchmarks speedups with `torch.compile` were not observed. 

A set of utility functions for MLP construction with Keras is also provided (i.e., handling input/output sizes for valid networks), replacing Sonnet.

## TensorFlow compatibility and test status

[![Tests](https://github.com/mylonasc/tf_gnns/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/mylonasc/tf_gnns/actions/workflows/tests.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/mylonasc/tf_gnns/main/docs/shields/coverage.json)](https://github.com/mylonasc/tf_gnns/blob/main/docs/shields/coverage.json)
[![Docs](https://github.com/mylonasc/tf_gnns/actions/workflows/docs-pages.yml/badge.svg?branch=main)](https://mylonasc.github.io/tf_gnns/)

| TensorFlow | TensorFlow Probability | Status |
|-----|-----|-----|
| 2.17.x | 0.24.x | <img alt="TF 2.17 TFP 0.24" src="https://img.shields.io/github/actions/workflow/status/mylonasc/tf_gnns/tests.yml?branch=main&job=test-tf217&label=TF%202.17%20%7C%20TFP%200.24"> |
| 2.18.x | 0.25.x | <img alt="TF 2.18 TFP 0.25" src="https://img.shields.io/github/actions/workflow/status/mylonasc/tf_gnns/tests.yml?branch=main&job=test-tf218&label=TF%202.18%20%7C%20TFP%200.25"> |
| 2.19.x | 0.25.x | <img alt="TF 2.19 TFP 0.25" src="https://img.shields.io/github/actions/workflow/status/mylonasc/tf_gnns/tests.yml?branch=main&job=test-tf219&label=TF%202.19%20%7C%20TFP%200.25"> |
| 2.20.x | 0.25.x | <img alt="TF 2.20 TFP 0.25" src="https://img.shields.io/github/actions/workflow/status/mylonasc/tf_gnns/tests.yml?branch=main&job=test-tf220&label=TF%202.20%20%7C%20TFP%200.25"> |
| 2.21.x | 0.25.x | <img alt="TF 2.21 TFP 0.25" src="https://img.shields.io/github/actions/workflow/status/mylonasc/tf_gnns/tests.yml?branch=main&job=test-tf221&label=TF%202.21%20%7C%20TFP%200.25"> |

The matrix above is validated by `scripts/run_tf_matrix_tests.sh` and in CI (`.github/workflows/tests.yml`).

## Installing `tf_gnns`
---
**NOTE**

The current tested matrix is TensorFlow `2.17` through `2.21` with the matching TensorFlow Probability versions shown above.

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

Run tests:
```
uv sync --group dev
uv run pytest -v
```

Run tests with coverage and update badge payload:
```
scripts/run_coverage.sh
```

Run compatibility tests across TensorFlow versions:
```
scripts/run_tf_matrix_tests.sh 2.17 2.18 2.19 2.20 2.21
```

## Execution and compilation

`tf_gnns` execution paths are eager by default so they can remain backend-portable with Keras 3.
If you are using the TensorFlow backend and want graph compilation, compile at the application level:

```python
import tensorflow as tf
from tf_gnns.models.graphnet import GraphNetMLP

model = GraphNetMLP(units=32, core_steps=2)

@tf.function
def train_step(graph_tensor_dict):
    with tf.GradientTape() as tape:
        out = model(graph_tensor_dict)
        loss = tf.reduce_mean(out["nodes"])  # example loss
    grads = tape.gradient(loss, model.trainable_variables)
    return loss, grads
```

This keeps library internals backend-agnostic while still allowing TensorFlow users to optimize execution.

### Torch backend note

For Keras 3 + Torch backend with Triton enabled, this repository is currently tested with:

- `torch==2.11.0`
- `triton==3.6.0` (installed as a dependency of torch 2.11.0)

Recommended setup:

```bash
pip install "torch==2.11.0"
KERAS_BACKEND=torch pytest -q tests
```

If you are using a different Torch/Triton combo and hit import-time crashes in
`triton` / `torch._dynamo`, pinning to the combination above is the first step.

Build the Docker test image for a specific TensorFlow version:
```
docker build --build-arg TENSORFLOW_VERSION=2.17 -t tf-gnns:test .
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

## Keras 3 + Torch backend example
This example demonstrates using the higher-level GraphNet constructs with Keras 3 configured for the PyTorch backend.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mylonasc/tf_gnns/blob/main/notebooks/04_keras3_torch_backend_graphnet.ipynb)

## GCN models
`tf_gnns` includes sparse GCN implementations for node-classification workloads:

- `SparseGCNConv`: low-level sparse graph convolution layer.
- `SparseGCN`: stacked sparse GCN model.
- `GCNv2`: tunedGNN-style high-level GCN stack with residual paths, normalization, and configurable dropout.

See the OGBN-Arxiv examples:

- `notebooks/06_gcn_ogbn_arxiv_tfgnns.ipynb` (tf_gnns GCN training workflow, including tunedGNN-style configuration)

## Performance
From initial tests, the performance of `tf_gnns` seems to be at least as good as `deepmind/graph_nets` when using tensor dictionaries.

# Publications using `tf_gnns`
The library has been used so far in the following publications:

\[1\] [Bayesian graph neural networks for strain-based crack localization](https://arxiv.org/abs/2012.06791) 

\[2\] [Remaining Useful Life Estimation Under Uncertainty with Causal GraphNets](https://arxiv.org/abs/2011.11740)

\[3\] [Relational VAE: A Continuous Latent Variable Model for Graph Structured Data](https://arxiv.org/abs/2106.16049)
