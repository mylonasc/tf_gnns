# Quickstart

Read this when creating a small `GraphTuple` and running one model.

## Rules

- Build object graphs with `Node`, `Edge`, and `Graph` when examples are small and human-readable.
- Convert to a batched `GraphTuple` with `make_graph_tuple_from_graph_list` before using high-level Keras layers.
- Call `GraphTuple.to_tensor_dict()` for layer wrappers such as `GraphNetMPNN_MLP`, `GraphNetMLP`, `GraphIndep`, and `SparseGCN`.
- Output tensor dictionaries preserve graph structure keys and replace transformed feature tensors.

## MPNN Example

```python
import tensorflow as tf
from tf_gnns import Edge, Graph, Node, make_graph_tuple_from_graph_list
from tf_gnns.models.graphnet import GraphNetMPNN_MLP

n0 = Node(tf.constant([[1.0, 0.0]], dtype=tf.float32))
n1 = Node(tf.constant([[0.0, 1.0]], dtype=tf.float32))
e01 = Edge(tf.constant([[0.5, 0.2]], dtype=tf.float32), n0, n1)
graph = Graph([n0, n1], [e01])

td = make_graph_tuple_from_graph_list([graph]).to_tensor_dict()
model = GraphNetMPNN_MLP(units=8, core_steps=1)
out = model(td)
assert set(td).issubset(out)
assert out["nodes"].shape[-1] == td["nodes"].shape[-1]
```
