# Graph Data

Read this when building `Node`, `Edge`, `Graph`, and `GraphTuple` inputs.

## How It Works

Graph data in tf_gnns has two representations: **object graphs** (individual `Node`/`Edge`/`Graph` instances) and **tensor dictionaries** (flat dicts consumed by model layers). `GraphTuple` is the batched representation that bridges the two.

The flow is: **object graphs → `make_graph_tuple_from_graph_list` → `GraphTuple` → `to_tensor_dict()` → model layers**. To unbatch, use `get_graph(i)`. Global features are added via `assign_global` and tracked per-node/edge via `update_reps_for_globals`.

**Key shapes** (one graph, D-dim features):

| Component | Object form | Tensor-dict key | Shape |
|---|---|---|---|
| Node features | `Node(tensor)` | `nodes` | [N, D_node] |
| Edge features | `Edge(tensor, src, dst)` | `edges` | [E, D_edge] |
| Connectivity | — | `senders`, `receivers` | [E] each |
| Counts | — | `n_nodes`, `n_edges`, `n_graphs` | scalar (per graph) |
| Global features | `Graph(..., global_attr=tensor)` | `global_attr` | [K, D_glob] |
| Global reps | — | `global_reps_for_nodes` | [N] (graph index per node) |
| Global reps | — | `global_reps_for_edges` | [E] (graph index per edge) |

## Decision Guide

- **Building a small graph for examples?** → Use `Node`, `Edge`, `Graph` object construction
- **Batching multiple graphs for model input?** → `make_graph_tuple_from_graph_list` → `to_tensor_dict()`
- **Adding global features to an existing batch?** → `gt.assign_global(tensor)` + `gt.update_reps_for_globals()`
- **Unbatching to inspect a single graph?** → `gt.get_graph(i)` returns an object `Graph`
- **Comparing graphs?** → `g.compare_connectivity(other)` (structural); field-wise via `to_tensor_dict()` + `tf.equal`

## Parameter Key Facts

| Parameter | Effect | Gotcha | Related |
|---|---|---|---|
| `Node(tensor)` | Creates a node with feature tensor | **Tensor must have rank ≥ 2.** Single features use `[[1.0, 0.0]]`, not `[1.0, 0.0]`. | `make_graph_tuple_from_graph_list` expects first dim 1 |
| `Edge(tensor, src, dst)` | Creates a directed edge | Auto-appends to `dst.incoming_edges` | — |
| `Graph(nodes, edges)` | Object graph | `NO_VALIDATION=True` skips validation | `global_attr` optional |
| `make_graph_tuple_from_graph_list([g1, g2])` | Batches object graphs | Node/edge tensors must have first dim 1 (object graphs) | `assign_global` after batching |
| `assign_global(tensor, check_shape=True)` | Adds global features | One row per graph; `check_shape=True` catches mistakes | `update_reps_for_globals` after assign |
| `get_subgraph_from_nodes(nodes, mode)` | Extracts subgraph | `"+from+to"` keeps edges with both endpoints kept; `"-from+to"` may raise `KeyError` | Prefer `"+from+to"` |

## Boundaries & Gotchas

1. **`GraphTuple.is_equal_by_value` crashes on vector-valued fields.** It uses Python `all()` on bool tensors. Compare via `to_tensor_dict()` field-by-field: `tf.reduce_all(tf.equal(a.nodes, b.nodes))`.
2. **`get_subgraph_from_nodes("-from+to")` raises `KeyError`.** When it must keep an edge (both endpoints excluded), it fails. Prefer `"+from+to"`.
3. **`make_graph_tuple_from_graph_list` drops `global_attr`.** Object-graph globals are not carried into the `GraphTuple`. Use `assign_global` after batching.
4. **Object node/edge tensors must have first dim 1.** The batcher expects `[1, feature_dim]` shapes for each object tensor.

## Minimal Skeleton

```python
import tensorflow as tf
from tf_gnns import Node, Edge, Graph, make_graph_tuple_from_graph_list

n0 = Node(tf.constant([[1.0, 0.0]], dtype=tf.float32))
n1 = Node(tf.constant([[0.0, 1.0]], dtype=tf.float32))
e = Edge(tf.constant([[0.5]], dtype=tf.float32), n0, n1)
gt = make_graph_tuple_from_graph_list([Graph([n0, n1], [e])])
td = gt.to_tensor_dict()
assert td["nodes"].shape[0] == 2
```

## Rules

- `Node(node_attr_tensor)` requires a tensor-like value with rank at least 2.
- `Edge(edge_attr_tensor, node_from, node_to)` is directed and appends itself to `node_to.incoming_edges`.
- `Graph(nodes, edges, global_attr=None)` stores object nodes, directed edges, and optional graph-level features.
- `make_graph_tuple_from_graph_list([...])` flattens multiple graphs into one `GraphTuple`.
- `GraphTuple.get_graph(i)` reconstructs an object `Graph`; compare connectivity with `Graph.compare_connectivity`.
- `GraphTuple.to_tensor_dict()` returns the tensor-dict format consumed by model layers.
- `Graph.get_subgraph_from_nodes(nodes, edge_trimming_mode="+from+to")` keeps only nodes in `nodes` and edges whose endpoints are both kept (`"+from+to"`); `"-from+to"` keeps edges whose endpoints are both excluded.
- `GraphTuple.assign_global(global_attr, check_shape=True)` validates that the global tensor has exactly one row per graph; pass `check_shape=True` to catch shape mistakes.
- After batching `K` graphs with `make_graph_tuple_from_graph_list([...])`, `n_nodes` has `K` entries and `n_nodes.sum() == td["nodes"].size`, and the same holds for `n_edges` and `edges`.
- `GraphTuple.is_equal_by_value` raises on any vector-valued field (use `to_tensor_dict()` field comparison); `get_subgraph_from_nodes` with `"-from+to"` raises `KeyError` when it must keep an edge (use `"+from+to"` instead).

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
- `Graph.copy()` / `Graph.is_equal_by_value(other)` / `Graph.compare_connectivity(other)`; `GraphTuple.copy()` / `GraphTuple.is_equal_by_value(other)`.
- `Node.get_state()` — the feature tensor of an object node; `Edge.edge_tensor` — the feature tensor of an object edge.
- `GraphTuple.update_reps_for_globals()` rebuilds `_global_reps_for_nodes` / `_global_reps_for_edges` from `n_nodes` / `n_edges` (each node/edge indexed by its graph, in order).

## Output Contract

- `to_tensor_dict()` output is always the key set above; `n_nodes` has one entry per graph and `sum(n_nodes) == nodes` row count (same for edges).
- `GraphTuple` objects also expose `.nodes/.edges/.senders/.receivers/.n_nodes/.n_edges` tensor attributes directly.
- Object-node features are read back with `Node.get_state()`; object-edge features with `Edge.edge_tensor` (these are the per-object read accessors for the object `Graph` form).

## Example Gallery

### Object Graph Example

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

### GraphTuple Example

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

### Multi-Graph Batching And Subgraph Example

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

### Copy, Equality, And Global Forwarding Vectors

```python
import tensorflow as tf
from tf_gnns import Edge, Graph, GraphTuple, Node, make_graph_tuple_from_graph_list

def make_g(v):
    n0 = Node(tf.constant([[float(v), 0.0]], dtype=tf.float32))
    n1 = Node(tf.constant([[0.0, float(v)]], dtype=tf.float32))
    n2 = Node(tf.constant([[float(v), float(v)]], dtype=tf.float32))
    e = Edge(tf.constant([[0.1 * v]], dtype=tf.float32), n0, n1)
    return Graph([n0, n1, n2], [e])

g1, g2 = make_g(1.0), make_g(2.0)
gc = g2.copy()
assert gc.is_equal_by_value(g2)
assert gc.compare_connectivity(g2)

gt = make_graph_tuple_from_graph_list([g1, g2])
td_before = gt.to_tensor_dict()

gt.assign_global(tf.constant([[1.0], [2.0]], dtype=tf.float32), check_shape=True)
gt.update_reps_for_globals()
td = gt.to_tensor_dict()
assert td["global_attr"].shape[0] == 2
assert td["global_reps_for_nodes"].numpy().tolist() == [0, 0, 0, 1, 1, 1]  # 3 nodes per graph
assert td["global_reps_for_edges"].numpy().tolist() == [0, 1]  # 1 edge per graph

# Compare GraphTuples field-wise (GraphTuple.is_equal_by_value raises for any
# vector-valued field - see boundary below).
same = GraphTuple(
    nodes=td["nodes"], edges=td["edges"], senders=td["senders"],
    receivers=td["receivers"], n_nodes=td["n_nodes"], n_edges=td["n_edges"],
    n_graphs=td["n_graphs"],
)
assert tf.reduce_all(tf.equal(same.nodes, gt.nodes)) and tf.reduce_all(tf.equal(same.edges, gt.edges))

# "-from+to" keeps an edge only when BOTH endpoints are excluded from the subset.
sub = g1.get_subgraph_from_nodes(g1.nodes[:1], edge_trimming_mode="-from+to")
assert len(sub.edges) == 0  # the only edge touches a kept node
assert isinstance(sub, Graph)
```
