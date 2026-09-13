# GraphNet Blocks

Read this when using `GraphNet` factories, MPNN layers, globals, and aggregators.

## Rules

- `make_mlp_graphnet_functions(...)` returns keyword arguments for `GraphNet(**kwargs)`.
- Use `make_mpnn_graphnet_noglobal_functions` or `GraphNetMPNN_MLP` when the graph has no global state.
- Use `make_full_graphnet_functions` or `GraphNetMLP` when node, edge, and global updates are required.
- Aggregation names are `mean`, `sum`, `max`, `min`, `mean_max`, `mean_max_min`, and `mean_max_min_sum`.
- Composite aggregations multiply message widths: `mean_max` -> 2x, `mean_max_min` -> 3x, `mean_max_min_sum` -> 4x.
- For global inputs, set `use_global_input=True` before setting `use_global_to_edge=True` or `use_global_to_node=True` (the reverse raises `ValueError`).
- `GraphNet.eval_tensor_dict(td)` is the direct tensor-dict path; `GraphNet.graph_tuple_eval(graph_tuple)` evaluates a batched `GraphTuple` directly; both return the same shapes for the same topology.

## Factory Example

```python
import tensorflow as tf
from tf_gnns import GraphNet
from tf_gnns.graphnet_utils import make_mlp_graphnet_functions

td = {
    "nodes": tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32),
    "edges": tf.constant([[0.5, 0.2]], dtype=tf.float32),
    "senders": tf.constant([0], dtype=tf.int32),
    "receivers": tf.constant([1], dtype=tf.int32),
    "n_nodes": tf.constant([2], dtype=tf.int32),
    "n_edges": tf.constant([1], dtype=tf.int32),
    "n_graphs": tf.constant(1, dtype=tf.int32),
    "global_attr": None,
    "global_reps_for_nodes": tf.constant([0, 0], dtype=tf.int32),
    "global_reps_for_edges": tf.constant([0], dtype=tf.int32),
}

gn_args = make_mlp_graphnet_functions(
    units=8,
    node_input_size=2,
    node_output_size=2,
    edge_input_size=2,
    edge_output_size=2,
    use_global_input=False,
    create_global_function=False,
    aggregation_function="mean",
)
out = GraphNet(**gn_args).eval_tensor_dict(td)
assert out["nodes"].shape[-1] == 2
```

## Layer Example

```python
import tensorflow as tf
from tf_gnns.models.graphnet import GraphIndep

td = {
    "nodes": tf.ones((3, 4)),
    "edges": tf.ones((2, 5)),
    "senders": tf.constant([0, 1]),
    "receivers": tf.constant([1, 2]),
    "n_nodes": tf.constant([3]),
    "n_edges": tf.constant([2]),
    "n_graphs": tf.constant(1),
    "global_attr": None,
    "global_reps_for_nodes": tf.constant([0, 0, 0]),
    "global_reps_for_edges": tf.constant([0, 0]),
}
layer = GraphIndep(units_out=6, create_global_function=False)
out = layer(td)
assert out["nodes"].shape[-1] == 6
assert out["edges"].shape[-1] == 6
```

## GraphNetMLP With Globals And Explicit Sizes

```python
import tensorflow as tf
from tf_gnns.models.graphnet import GraphNetMLP

td = {
    "nodes": tf.ones((4, 3), dtype=tf.float32),
    "edges": tf.ones((5, 4), dtype=tf.float32),
    "senders": tf.constant([0, 1, 2, 3, 0], dtype=tf.int32),
    "receivers": tf.constant([1, 2, 3, 0, 2], dtype=tf.int32),
    "n_nodes": tf.constant([4], dtype=tf.int32),
    "n_edges": tf.constant([5], dtype=tf.int32),
    "n_graphs": tf.constant(1, dtype=tf.int32),
    "global_attr": tf.ones((1, 2), dtype=tf.float32),
    "global_reps_for_nodes": tf.constant([0, 0, 0, 0], dtype=tf.int32),
    "global_reps_for_edges": tf.constant([0, 0, 0, 0, 0], dtype=tf.int32),
}
model = GraphNetMLP(
    units=16, core_steps=2, recurrent=False, residual=True,
    node_output_size=7, edge_output_size=5, global_output_size=3,
)
out = model(td)
assert out["nodes"].shape == (4, 7)
assert out["edges"].shape == (5, 5)
assert out["global_attr"].shape == (1, 3)
```

## Factory With Globals And Tensor-Dict/GraphTuple Consistency

```python
import tensorflow as tf
from tf_gnns import GraphTuple
from tf_gnns.graphnet_utils import GraphNet, make_mlp_graphnet_functions

gedges = {
    "nodes": tf.ones((4, 3), dtype=tf.float32),
    "edges": tf.ones((4, 4), dtype=tf.float32),
    "senders": tf.constant([0, 1, 2, 3], dtype=tf.int32),
    "receivers": tf.constant([1, 2, 3, 0], dtype=tf.int32),
    "n_nodes": tf.constant([2, 2], dtype=tf.int32),
    "n_edges": tf.constant([2, 2], dtype=tf.int32),
    "n_graphs": tf.constant(2, dtype=tf.int32),
    "global_attr": tf.ones((2, 2), dtype=tf.float32),
    "global_reps_for_nodes": tf.constant([0, 0, 1, 1], dtype=tf.int32),
    "global_reps_for_edges": tf.constant([0, 0, 1, 1], dtype=tf.int32),
}

gn_args = make_mlp_graphnet_functions(
    units=8,
    node_input_size=3,
    node_output_size=6,
    edge_input_size=4,
    edge_output_size=6,
    global_input_size=2,
    global_output_size=3,
    create_global_function=True,
    use_global_input=True,
    use_global_to_edge=True,
    use_global_to_node=True,
    aggregation_function="mean_max_min_sum",
)
net = GraphNet(**gn_args)
out_td = net.eval_tensor_dict(gedges.copy())

gt = GraphTuple(
    nodes=gedges["nodes"], edges=gedges["edges"],
    senders=gedges["senders"], receivers=gedges["receivers"],
    n_nodes=gedges["n_nodes"], n_edges=gedges["n_edges"],
    n_graphs=gedges["n_graphs"], global_attr=gedges["global_attr"],
)
out_gt = net.graph_tuple_eval(gt.copy())
assert out_td["nodes"].shape == out_gt.nodes.shape
```
