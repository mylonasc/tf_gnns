Architecture
============

``tf_gnns`` is organized around one central idea: represent graph structure with
plain tensors, express GNN computation as Keras layers and Keras models, and keep
the primitive tensor operations routed through ``keras.ops``. This allows the
same model code to run on the TensorFlow, JAX, and Torch backends supported by
Keras 3.

The architecture below is an idealized view of the core runtime. It omits
documentation, examples, tests, and benchmark adapters so the separation between
model APIs, GraphNet construction, graph representations, and backend tensor
operations is explicit.

.. graphviz:: figures/architecture.dot
   :caption: ``tf_gnns`` module architecture and data-flow boundaries.

The rendered PDF version is available at ``docs/figures/architecture.pdf`` for
offline review.

Core Design
-----------

The library separates graph neural network computation into four layers.

1. User-facing model APIs provide Keras layers for common GNN families.
2. GraphNet construction utilities define message-passing semantics and create
   learnable update functions.
3. Graph data structures encode graph topology and batch structure in tensors.
4. Backend operations provide a small compatibility layer over ``keras.ops``.

The two primary model families are GraphNet-style message-passing networks and
sparse GCNs. They use different compute implementations, but they share the same
runtime graph representation: a graph tensor dictionary containing node features,
edge features, sender indices, receiver indices, graph counts, and optional
global features.

Layer 1: User-Facing Model APIs
-------------------------------

The high-level APIs live in ``tf_gnns.models`` and are regular Keras layers.
They are the entry points most users compose into larger Keras models.

``tf_gnns.models.graphnet`` provides GraphNet/MPNN layers:

``GraphNetMLP``
   An encode-process-decode GraphNet with edge, node, and global states. It is
   suitable when graph-level attributes are part of the model state or output.

``GraphNetMPNN_MLP``
   An encode-process-decode MPNN that omits global updates. It is useful for
   common message-passing workloads over node and edge states.

``GNCellMLP``
   A single GraphNet processing cell implemented with MLP update functions.

``GraphIndep``
   A graph-independent transform that updates node, edge, and optionally global
   features without message passing.

``tf_gnns.models.gcn`` provides sparse GCN layers:

``SparseGCNConv``
   A single sparse graph convolution. It gathers source node states with
   ``senders``, aggregates them into destination nodes with ``receivers``, and
   applies a trainable linear transform with optional self-loops,
   normalization, bias, activation, and normalization layers.

``SparseGCN``
   A stack of ``SparseGCNConv`` layers with dropout and optional residual
   connections.

``GCNv2``
   A higher-level tunedGNN-style GCN stack with repeated hidden convolutions,
   residual projection, dropout, optional shortcut accumulation, and a dense
   prediction head.

Both families accept and return graph tensor dictionaries. This is the main
contract that lets higher-level Keras models wrap GNN blocks while preserving
graph topology.

Layer 2: GraphNet Construction and Message Passing
--------------------------------------------------

GraphNet-style models are built from ``tf_gnns.graphnet_utils.GraphNet``. A
``GraphNet`` object represents one message-passing block. It owns three kinds of
functions:

``edge_function``
   Computes updated edge/message states from any combination of edge state,
   sender node state, receiver node state, and global state.

``node_function``
   Computes updated node states from node state, aggregated incoming edge
   messages, and optionally global state.

``global_function``
   Computes updated graph-level features from global state and aggregated node
   and edge states.

The main runtime method is ``GraphNet.eval_tensor_dict()``. It evaluates one
GraphNet block over the tensor-dict graph representation:

1. ``edge_block()`` gathers the sender and receiver node rows referenced by each
   edge, gathers repeated global state when requested, and calls the Keras edge
   update model.
2. ``node_block()`` aggregates updated edge states by ``receivers`` using a
   segment reducer, then calls the Keras node update model.
3. ``global_block()`` optionally aggregates node and edge states by graph id and
   calls the Keras global update model.

The helper factories in ``graphnet_utils`` create the Keras models and reducers
used by ``GraphNet``:

``make_edge_mlp()``, ``make_node_mlp()``, and ``make_global_mlp()``
   Construct named-input Keras MLPs for the edge, node, and global update
   functions.

``make_mlp_graphnet_functions()``
   Creates a full dictionary of constructor arguments for ``GraphNet``, including
   edge/node/global functions and aggregation functions.

``make_full_graphnet_functions()``
   Builds a GraphNet block that consumes and updates global state.

``make_mpnn_graphnet_noglobal_functions()``
   Builds an MPNN block with edge-to-node message passing but no global state
   updates.

``make_graph_indep_graphnet_functions()``
   Builds a graph-independent block that transforms features without message
   passing.

Aggregators are created by ``_aggregation_function_factory()``. The simple modes
``mean``, ``sum``, ``max``, and ``min`` map directly to backend segment
reductions. Compound modes such as ``mean_max`` and ``mean_max_min_sum`` compute
multiple reductions and concatenate their outputs, increasing the information
available to the node or global update function.

Layer 3: Graph Representation
-----------------------------

The graph representation layer has an ergonomic object form and a computational
tensor form.

``Node``, ``Edge``, and ``Graph``
   These classes represent object graphs. They are convenient for graph
   construction, copying, connectivity checks, and subgraph extraction. Edges
   connect source and destination ``Node`` instances, and each edge carries its
   own edge feature tensor.

``GraphTuple``
   This is the flattened batched representation. Instead of storing Python
   object references, it stores tensors and index arrays. All node features are
   stacked in ``nodes``; all edge features are stacked in ``edges``; topology is
   encoded by ``senders`` and ``receivers``. Batch boundaries are stored in
   ``n_nodes`` and ``n_edges``. Global feature broadcasting and aggregation use
   ``global_reps_for_nodes`` and ``global_reps_for_edges``.

``GraphTuple.to_tensor_dict()``
   Converts a ``GraphTuple`` into the dictionary format consumed by GraphNet and
   GCN layers. This dictionary is the canonical runtime interface.

The graph tensor dictionary contains these core keys:

``nodes``
   Node feature tensor with shape ``[num_nodes, node_feature_size]``.

``edges``
   Edge feature tensor with shape ``[num_edges, edge_feature_size]``.

``senders`` and ``receivers``
   Integer index tensors of shape ``[num_edges]``. ``senders[i]`` is the source
   node for edge ``i`` and ``receivers[i]`` is the destination node.

``n_nodes`` and ``n_edges``
   Per-graph counts for batched graph processing.

``global_attr``
   Optional graph-level feature tensor with shape
   ``[num_graphs, global_feature_size]``.

``global_reps_for_nodes`` and ``global_reps_for_edges``
   Integer index tensors mapping each node or edge row to its graph id. These are
   used to gather graph-level state to nodes/edges and to aggregate node/edge
   state back to graph-level outputs.

``n_graphs``
   Number of graphs represented by the flattened batch.

The helpers in ``tf_gnns.lib.gt_ops`` operate on this dictionary while preserving
topology. For example, ``_copy_structure()`` copies only structural fields,
``_add_gt()`` adds feature tensors from two compatible graphs, and
``_concat_tensordicts()`` concatenates feature channels while keeping the same
connectivity.

Layer 4: Backend Tensor Primitives
----------------------------------

Backend independence is provided by ``tf_gnns.backend_ops``. This module is a
small facade over ``keras.ops``. Core library code uses this facade instead of
calling TensorFlow, JAX, or Torch APIs directly.

Important primitive operations include:

- ``convert_to_tensor()`` converts Python, NumPy, or backend-native inputs into
  tensors for the active Keras backend.
- ``gather()`` selects rows for sender nodes, receiver nodes, or repeated global
  state.
- ``concat()`` joins feature tensors, especially for compound aggregations and
  MLP inputs.
- ``segment_sum()``, ``segment_mean()``, ``segment_max_or_zero()``, and
  ``segment_min_or_zero()`` implement sparse graph aggregation by segment id.
  These operations are the primitive foundation for edge-to-node and
  node/edge-to-global message passing.

Because these primitives delegate to ``keras.ops``, the same high-level model can
run with ``KERAS_BACKEND=tensorflow``, ``KERAS_BACKEND=jax``, or
``KERAS_BACKEND=torch`` when the corresponding backend is installed.

GraphNet/MPNN Runtime Flow
--------------------------

A typical ``GraphNetMPNN_MLP`` call follows this path:

1. The user passes a graph tensor dictionary to the Keras layer.
2. The layer applies a graph-independent encoder built from ``GraphNet`` blocks.
3. Each processor step calls ``GraphNet.eval_tensor_dict()``.
4. The edge block gathers source and destination node states and computes edge
   messages.
5. The node block aggregates incoming edge messages with backend segment
   operations and computes updated node states.
6. Residual connections optionally add the processor output back into the latent
   graph state.
7. A graph-independent decoder maps latent node and edge states to output feature
   sizes.

``GraphNetMLP`` follows the same encode-process-decode pattern, but includes
global state in the encoder, processor, and decoder when ``global_attr`` is part
of the input graph.

Sparse GCN Runtime Flow
-----------------------

The sparse GCN path is more direct. It does not instantiate ``GraphNet`` blocks.
Instead, ``SparseGCNConv.call()`` implements the sparse convolution explicitly:

1. Read ``nodes``, ``senders``, ``receivers``, and optional ``edge_weights`` from
   the graph tensor dictionary.
2. Optionally append self-loops.
3. Compute symmetric degree normalization weights.
4. Gather source node states using the edge source indices.
5. Scale gathered source states by normalized edge weights.
6. Aggregate scaled messages into destination nodes with ``segment_sum``.
7. Apply trainable linear transformations, optional root transform, bias,
   normalization, and activation.
8. Return a copied graph tensor dictionary with updated ``nodes`` and preserved
   topology.

``SparseGCN`` and ``GCNv2`` stack this convolutional primitive into deeper Keras
layers, adding dropout, residual paths, and prediction heads as needed.

Why This Is Backend Independent
-------------------------------

``tf_gnns`` keeps the backend-dependent part of GNN computation small. Model
composition is handled by Keras layers and Keras models. Graph topology is stored
as ordinary tensors and index arrays. Message passing is expressed through a
small set of primitive tensor operations: gather, concatenate, matrix multiply,
and segment reduction.

Keras 3 supplies backend-specific implementations of those primitives through
``keras.ops``. ``tf_gnns.backend_ops`` centralizes the details needed by graph
computation, including dtype handling and zero semantics for empty segment
reductions. As a result, the higher-level GraphNet and GCN code can remain
backend-agnostic while still using efficient sparse tensor-style computation.
