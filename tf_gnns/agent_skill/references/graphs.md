# Graph Data

Read this when building `Node`, `Edge`, `Graph`, and `GraphTuple` inputs.

## Rules

- `Node(node_attr_tensor)` requires a tensor-like value with rank at least 2.
- `Edge(edge_attr_tensor, node_from, node_to)` is directed and appends itself to `node_to.incoming_edges`.
- `Graph(nodes, edges, global_attr=None)` stores object nodes, directed edges, and optional graph-level features.
- `make_graph_tuple_from_graph_list([...])` flattens multiple graphs into one `GraphTuple`.
- `GraphTuple.get_graph(i)` reconstructs an object `Graph`; compare connectivity with `Graph.compare_connectivity`.
- `GraphTuple.to_tensor_dict()` returns the tensor-dict format consumed by model layers.
- `Graph.get_subgraph_from_nodes(nodes, edge_trimming_mode="+from+to")` keeps only nodes in `nodes` and edges whose endpoints are both kept (`"+from+to"`); `"-from+to"` keeps edges whose endpoints are both excluded.
- `GraphTuple.assign_global(global_attr, check_shape=True)` validates that the global tensor has exactly one row per graph; pass `check_shape=True` to catch shape mistakes.
- After batching `K` graphs with `make_graph_tuple_from_graph_list(node_features, edge_features, ...)`, `n_nodes` has `K` entries and `n_nodes.sum() == td["nodes"].size`, and the same holds for `n_edges` and `edges`.

## Object Graph Example

```python
import tensorflow as tf
from tf_gnns import Edge, Graph, Node

n0 = Node(tf.constant([[1.0, 2.0]], dtype=tf.float32))
n1 = Node(tf.constant([[3.0, 4.0]], dtype=tf.float32))
edge = Edge(tf.constant([[0.1, 0.2]], dtype=tf.float32), n0, n1)
graph = Graph([n0, n1], [edge])

copy = graph.copy()
assert graph.compare_connectivity(copy)
```

## GraphTuple Example

```python
import tensorflow as tf
from tf_gnns import Edge, Graph, GraphTuple, Node, make_graph_tuple_from_graph_list

n0 = Node(tf.constant([[1.0, 2.0]], dtype=tf.float32))
n1 = Node(tf.constant([[3.0, 4.0]], dtype=tf.float32))
edge = Edge(tf.constant([[0.1, 0.2]], dtype=tf.float32), n0, n1)
gt = make_graph_tuple_from_graph_list([Graph([n0, n1], [edge])])
gt.assign_global(tf.constant([[5.0, 6.0]], dtype=tf.float32), check_shape=True)
td = gt.to_tensor_dict()

assert list(td) == [
    "edges", "nodes", "senders", "receivers", "n_edges", "n_nodes",
    "n_graphs", "global_attr", "global_reps_for_edges", "global_reps_for_nodes",
]
assert td["senders"].shape[0] == td["receivers"].shape[0]

manual = GraphTuple(
    nodes=tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32),
    edges=tf.constant([[0.1, 0.2]], dtype=tf.float32),
    senders=tf.constant([0], dtype=tf.int32),
    receivers=tf.constant([1], dtype=tf.int32),
    n_nodes=tf.constant([2], dtype=tf.int32),
    n_edges=tf.constant([1], dtype=tf.int32),
)
assert manual.n_graphs == 1
```

## Multi-Graph Batching And Subgraph Example

```python
import tensorflow as tf
from tf_gnns import Edge, Graph, Node, make_graph_tuple_from_graph_list

def make_graph(value):
    n0 = Node(tf.constant([[float(value), 0.0]], dtype=tf.float32))
    n1 = Node(tf.constant([[0.0, float(value)]], dtype=tf.float32))
    e = Edge(tf.constant([[0.1 * value]], dtype=tf.float32), n0, n1)
    return Graph([n0, n1], [e])

gt = make_graph_tuple_from_graph_list([make_graph(1.0), make_graph(2.0)])
gt.assign_global(tf.constant([[1.0], [2.0]], dtype=tf.float32), check_shape=True)
td = gt.to_tensor_dict()

assert td["nodes"].shape[0] == int(td["n_nodes"].numpy().sum())
assert td["edges"].shape[0] == int(td["n_edges"].numpy().sum())
assert td["global_attr"].shape[0] == 2  # one row per graph

g0 = gt.get_graph(0)
assert g0.compare_connectivity(make_graph(1.0))

two_nodes = g0.get_subgraph_from_nodes(list(g0.nodes[:1]), edge_trimming_mode="+from+to")
```

## API Signatures (authoritative, no need to read source)

- `Node(node_attr_tensor)` — tensor-like with rank at least 2.
- `Edge(edge_attr_tensor, node_from, node_to)` — auto-appends to `node_to.incoming_edges`.
- `Graph(nodes, edges, global_attr=None, NO_VALIDATION=True)`.
- `GraphTuple(nodes, edges, senders, receivers, n_nodes, n_edges, global_attr=None, global_reps_for_nodes=None, global_reps_for_edges=None, n_graphs=None)`.
- `make_graph_tuple_from_graph_list(list_of_graphs)` batches object graphs into one `GraphTuple`; object node/edge tensors must have first dimension `1`.
- `GraphTuple.to_tensor_dict()` -> dict with keys `nodes, edges, senders, receivers, n_nodes, n_edges, n_graphs, global_attr, global_reps_for_edges, global_reps_for_nodes`.
- `GraphTuple.get_graph(i)` -> object `Graph`; `Graph.compare_connectivity(other)` -> bool.
- `GraphTuple.assign_global(global_attr, check_shape=False)` — with `check_shape=True` it raises if rows != number of graphs.
- `Graph.get_subgraph_from_nodes(nodes, edge_trimming_mode="+from+to")` — `"+from+to"` keeps edges with both endpoints kept; `"-from+to"` keeps edges with both endpoints excluded.

## Output Contract

- `to_tensor_dict()` output is always the key set above; `n_nodes` has one entry per graph and `sum(n_nodes) == nodes` row count (same for edges).
- `GraphTuple` objects also expose `.nodes/.edges/.senders/.receivers/.n_nodes/.n_edges` tensor attributes directly.
