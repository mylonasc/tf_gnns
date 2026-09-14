# GraphNet Blocks

Read this when using `GraphNet` factories, MPNN layers, globals, and aggregators.

## How It Works

GraphNet blocks process graph data through three function families — **edge**, **node**, and **global** — that consume and produce specific keys in a tensor dictionary. Factories (`make_*_graphnet_functions`) build the MLPs and return keyword arguments for `GraphNet(**kwargs)`. Class wrappers (`GraphNetMLP`, `GraphNetMPNN_MLP`) do the same with additional options (recurrent, residual, core_steps).

The data flow is: **input tensor dict → edge MLP (per edge) → aggregate edges per receiver node → concat aggregated message with node features → node MLP → optional global MLP**. Aggregation concatenates edge outputs, so composite modes (`mean_max`, etc.) multiply the message width fed to the node update.

Object-graph dispatch (`graph_eval`) works only for no-global blocks, because `make_graph_tuple_from_graph_list` drops `global_attr`. Global blocks must use `eval_tensor_dict` or `graph_tuple_eval`.

**Shape table** (integer `units`, one graph, output sizes set explicitly):

| Input tensor-dict keys | Output tensor-dict keys | Output shape |
|---|---|---|
| `nodes` [N, D_node] | `nodes` [N, node_output_size] | node_output_size |
| `edges` [E, D_edge] | `edges` [E, edge_output_size] | edge_output_size |
| `global_attr` [1, D_glob] | `global_attr` [1, global_output_size] | global_output_size |
| (omitted) | `global_attr` not present | — |
| `senders`, `receivers`, `n_nodes`, `n_edges`, `n_graphs` | same, unchanged | — |

## Decision Guide

- **Need node, edge, AND global updates?** → `make_full_graphnet_functions` or `GraphNetMLP` (the latter **requires** `global_attr` + repetition keys in the input dict)
- **Need message-passing only (no global state)?** → `make_mpnn_graphnet_noglobal_functions` or `GraphNetMPNN_MLP`
- **Need local per-node/edge updates only?** → `make_graph_indep_graphnet_functions` or `GraphIndep`
- **Single one-step block as a Keras layer?** → `GNCellMLP(gn_mlp_units=..., core_size=...)`
- **Faithful save/load roundtrip?** → `make_graph_indep_graphnet_functions(..., create_global_function=False, use_global_input=False)`

## Parameter Key Facts

| Parameter | Effect | Gotcha | Related |
|---|---|---|---|
| `units` | Hidden layer width (int) | **Pass as integer, not a list.** A list silently uses `units[-1]` as output and ignores all requested output sizes. | `make_mlp`, `make_mlp_graphnet_functions` |
| `node_output_size` | Width of node features in output | Defaults to input width when unset. | `node_or_core_output_size` (factories) |
| `edge_output_size` | Width of edge features in output | Defaults to input width when unset. Message width = `multiplier * edge_output_size`. | `aggregation_function` |
| `global_output_size` | Width of global features in output | Defaults to input width when unset. | `global_function` not serialized by `save()` |
| `recurrent` | Reuse one process block (True) or build per step (False) | Default: `False`. Recurrent has fewer weights at same core_steps. | `all_weights` only after first call |
| `residual` | Add block output to input (True) or replace (False) | Default: `True`. Both produce same shapes. | — |
| `aggregation_function` | How edges are aggregated per receiver | `mean/sum/max/min` → 1x message; `mean_max` → 2x; `mean_max_min` → 3x; `mean_max_min_sum` → 4x | `edge_state_agg` input width |
| `create_global_function` | Whether to build a global update MLP | `save()` does not serialize this; reload loses global updates | `use_global_input` must be True |
| `core_steps` | Number of message-passing iterations | Interacts with `recurrent` for weight count | `GraphNetMPNN_MLP`, `GraphNetMLP` |

## Boundaries & Gotchas

1. **`make_mlp(units=[...])` silently ignores output sizes.** A list of units makes every MLP end at `units[-1]`. Use `units=<int>` so hidden layers share a width and a final layer appends the exact requested output size.
2. **`graph_eval` on a full (global) block raises.** `make_graph_tuple_from_graph_list` drops `graph.global_attr`. Use `graph_eval` only with no-global MPNN blocks; drive global blocks through `eval_tensor_dict`/`graph_tuple_eval`.
3. **`save()` drops `global_function`.** Reloaded blocks lose global updates and cannot re-run message passing (segment-aggregator tuples not serialized). For a faithful roundtrip, save/load a no-global graph-independent block.
4. **`all_weights` is empty before the first call.** Call the layer (or pass `edge_input_size`/`node_input_size` to the constructor) before inspecting `all_weights`.
5. **`GraphIndep(..., create_global_function=...)` is not accepted.** No-global mode is selected by the input shape (no `global_attr` key), not by a kwarg.
6. **`GraphNetMLP` always builds a global block and fails without `global_attr`.** Pass a full dict (with `global_attr` plus `global_reps_for_nodes`/`global_reps_for_edges`). For a no-global pipeline use `GraphNetMPNN_MLP`.
7. **`tf.concat` requires `axis`.** `tf.concat([a, b])` raises TypeError in TF 2.21; use `tf.concat([a, b], axis=0)`.
8. **`tf.is_finite` → `tf.math.is_finite`** in TF 2.21.

## Minimal Skeleton

```python
import tensorflow as tf
from tf_gnns import GraphNet
from tf_gnns.graphnet_utils import make_mlp_graphnet_functions

td = {
    "nodes": tf.constant([[1.0, 0.0]], dtype=tf.float32),
    "edges": tf.constant([[0.5]], dtype=tf.float32),
    "senders": tf.constant([0], dtype=tf.int32),
    "receivers": tf.constant([0], dtype=tf.int32),
    "n_nodes": tf.constant([1], dtype=tf.int32),
    "n_edges": tf.constant([1], dtype=tf.int32),
    "n_graphs": tf.constant(1, dtype=tf.int32),
}
fns = make_mlp_graphnet_functions(units=8, node_input_size=2, node_output_size=4,
                                   edge_input_size=1, edge_output_size=3)
out = GraphNet(**fns).eval_tensor_dict(td)
assert out["nodes"].shape == (1, 4)
```

## Rules

- `make_mlp_graphnet_functions(...)` returns keyword arguments for `GraphNet(**kwargs)`.
- Use `make_mpnn_graphnet_noglobal_functions` or `GraphNetMPNN_MLP` when the graph has no global state.
- Use `make_full_graphnet_functions` or `GraphNetMLP` when node, edge, and global updates are required.
- Aggregation names are `mean`, `sum`, `max`, `min`, `mean_max`, `mean_max_min`, and `mean_max_min_sum`.
- Composite aggregations multiply message widths: `mean_max` -> 2x, `mean_max_min` -> 3x, `mean_max_min_sum` -> 4x.
- For global inputs, set `use_global_input=True` before setting `use_global_to_edge=True` or `use_global_to_node=True` (the reverse raises `ValueError`).
- `GraphNet.eval_tensor_dict(td)` is the direct tensor-dict path; `GraphNet.graph_tuple_eval(graph_tuple)` evaluates a batched `GraphTuple` directly; both return the same shapes for the same topology.
- `GraphNet.graph_eval(graph)` evaluates a single object `Graph` and returns a `Graph` with the same connectivity and updated features; `GraphNet.__call__` dispatches to it for object graphs.
- **Object-graph dispatch cannot carry global state**: `make_graph_tuple_from_graph_list` drops `graph.global_attr`, so `graph_eval` on a full (global) block raises. Use `graph_eval` only with no-global MPNN blocks; drive global blocks through `eval_tensor_dict`/`graph_tuple_eval`.
- **Pass `units` as an integer** (e.g. `units=8`), not a list: a list of units makes every MLP end at `units[-1]` and silently *ignores* the requested output sizes, producing blocks that fail when evaluated. Integer `units` generates hidden layers of that width and appends a final layer with the exact requested output size.
- `make_full_graphnet_functions` builds a full edge+node+global block (global routed into edge and node inputs); `make_mpnn_graphnet_noglobal_functions` builds an MPNN block with no global input or output.
- `GraphNet.save(path)` writes `node_function`, `edge_aggregation_function`, and `edge_function` as `.keras` files but **not** `global_function`. A reloaded block loses global updates, and message-passing blocks cannot be re-evaluated after load (segment-aggregator tuples are not serialized). For a faithful numerical roundtrip (outputs and weights identical), save/load a no-global graph-independent block: `make_graph_indep_graphnet_functions(..., create_global_function=False, use_global_input=False)`.
- `recurrent=True` reuses one process block across all `core_steps`; `recurrent=False` builds one block per step. At equal `core_steps`, `recurrent=True` has strictly fewer trainable weights (check `layer.all_weights`).
- `residual=True` adds each process block's output to its input; `residual=False` replaces the input. Both produce the same shapes.
- `GNCellMLP(gn_mlp_units, core_size, ...)` is a single one-step GraphNet block layer: it builds a full block when the input dict has `global_attr`, otherwise an MPNN no-global block.

## API Signatures (authoritative, no need to read source)

- `GraphNetMLP(units=32, core_units=None, core_size=None, gi_units=None, core_steps=1, edge_input_size=None, node_input_size=None, global_input_size=None, edge_output_size=None, node_output_size=None, global_output_size=None, recurrent=False, residual=True, aggregation_function="mean")`
- `GraphNetMPNN_MLP(units=32, core_units=None, core_size=None, gi_units=None, core_steps=1, edge_input_size=None, node_input_size=None, edge_output_size=None, node_output_size=None, recurrent=False, residual=True, aggregation_function="mean")`
- `GraphIndep(units_out, gn_mlp_units=[], node_output_size=None, edge_output_size=None, global_output_size=None, activation="relu", **kwargs)` — default output width is `units_out` for nodes, edges, and globals. No `create_global_function` kwarg; the no-global path is selected by the input shape (no `global_attr` key).
- `make_mlp_graphnet_functions(units, node_input_size, node_output_size, edge_input_size=None, edge_output_size=None, create_global_function=False, global_input_size=None, global_output_size=None, use_global_input=False, use_global_to_edge=False, use_global_to_node=False, node_mlp_use_edge_state_agg_input=True, graph_indep=False, message_size="auto", aggregation_function="mean", node_to_global_aggr_fn=None, edge_to_global_aggr_fn=None, activation="relu", activate_last_layer=False, **kwargs)`
- `make_full_graphnet_functions(units, node_or_core_input_size, node_or_core_output_size=None, edge_input_size=None, edge_output_size=None, global_input_size=None, global_output_size=None, aggregation_function="mean", **kwargs)`
- `make_graph_indep_graphnet_functions(units, node_or_core_input_size, node_or_core_output_size=None, edge_input_size=None, edge_output_size=None, global_input_size=None, global_output_size=None, aggregation_function="mean", create_global_function=True, use_global_input=True, **kwargs)`
- `make_mpnn_graphnet_noglobal_functions(units, node_or_core_input_size, node_or_core_output_size=None, edge_input_size=None, edge_output_size=None, aggregation_function="mean", **kwargs)`
- `GraphNet.eval_tensor_dict(td)` and `GraphNet.graph_tuple_eval(graph_tuple)` are the two evaluation paths; both return the same shapes for the same topology.
- `GraphNet.graph_eval(graph)` evaluates a single object `Graph` and returns a `Graph`; `GraphNet.save(path)`, `GraphNet.make_from_path(path)`, `GraphNet.load_graph_functions(path)`, and instance `GraphNet.load(path)` handle serialization.
- `GNCellMLP(gn_mlp_units, core_size=None, node_output_size=None, edge_output_size=None, global_output_size=None, aggregation_function="mean")` — single one-step GraphNet block layer.

## Output Contract

- Layer subclasses (`GraphNetMLP`, `GraphNetMPNN_MLP`, `GraphIndep`) and `eval_tensor_dict`/`graph_tuple_eval` return a dictionary that preserves the input structure keys and replaces feature tensors.
- The `global_attr` key is present in the output only when a global update function exists:
  - `GraphNetMPNN_MLP` is the no-global layer: pass a dict without `global_attr` and no `global_attr` output key is produced.
  - `GraphNetMLP` always builds a global block and requires `global_attr` (plus `global_reps_for_nodes`/`global_reps_for_edges`) in the input; its output `global_attr` width is `global_output_size`.
  - `GraphIndep` produces no `global_attr` output key when there is no global path: pass a dict without `global_attr` (or build on `{"nodes", "edges"}` shapes first and keep `global_attr` as `None`).
- Node/edge/global output widths follow `node_output_size`/`edge_output_size`/`global_output_size`; when unset they default to the input widths or `units_out`. This holds only for **integer `units`**; a list of units forces every MLP output to `units[-1]` and ignores the requested sizes.
- `layer.all_weights` is populated only after the layer is built (first call, or when `edge_input_size`/`node_input_size` are passed to the constructor) — call the layer before inspecting it.
- Message width fed to the node update is `multiplier * edge_output_size` where `mean/sum/max/min` -> 1x, `mean_max` -> 2x, `mean_max_min` -> 3x, `mean_max_min_sum` -> 4x; observable as the `edge_state_agg` input channel count of `node_function`.
- `graph_eval` works on object graphs only for no-global blocks (batching drops `global_attr`); global blocks must run via `eval_tensor_dict`/`graph_tuple_eval`.
- `graph_eval` returns an object `Graph`; read its node features back with `Node.get_state()` and edge features with `Edge.edge_tensor`.
- A reloaded block (`make_from_path`/`load`) keeps only the serialized functions: global updates are dropped and message-passing evaluation is not supported; roundtrip-correct results require a no-global graph-independent block.

## Example Gallery

### Factory Example

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

### Layer Example

```python
import tensorflow as tf
from tf_gnns.models.graphnet import GraphIndep

# No-global tensor dictionary: no "global_attr" key at all -> GraphIndep
# selects the graph-independent, no-global path automatically.
td = {
    "nodes": tf.ones((3, 4)),
    "edges": tf.ones((2, 5)),
    "senders": tf.constant([0, 1]),
    "receivers": tf.constant([1, 2]),
    "n_nodes": tf.constant([3]),
    "n_edges": tf.constant([2]),
    "n_graphs": tf.constant(1),
}
layer = GraphIndep(units_out=6)
out = layer(td)
assert out["nodes"].shape[-1] == 6
assert out["edges"].shape[-1] == 6
assert "global_attr" not in out  # no global output on the no-global path
```

### GraphNetMLP With Globals And Explicit Sizes

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

### Factory With Globals And Tensor-Dict/GraphTuple Consistency

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

### Direct Factory Uses And Object-Graph Dispatch

```python
import tensorflow as tf
from tf_gnns import Edge, Graph, Node
from tf_gnns.graphnet_utils import (
    GraphNet,
    make_full_graphnet_functions,
    make_mpnn_graphnet_noglobal_functions,
)

td_glob = {
    "nodes": tf.ones((3, 3), dtype=tf.float32),
    "edges": tf.ones((3, 4), dtype=tf.float32),
    "senders": tf.constant([0, 1, 2], dtype=tf.int32),
    "receivers": tf.constant([1, 2, 0], dtype=tf.int32),
    "n_nodes": tf.constant([3], dtype=tf.int32),
    "n_edges": tf.constant([3], dtype=tf.int32),
    "n_graphs": tf.constant(1, dtype=tf.int32),
    "global_attr": tf.ones((1, 2), dtype=tf.float32),
    "global_reps_for_nodes": tf.constant([0, 0, 0], dtype=tf.int32),
    "global_reps_for_edges": tf.constant([0, 0, 0], dtype=tf.int32),
}

full = make_full_graphnet_functions(
    units=8, node_or_core_input_size=3, node_or_core_output_size=6,
    edge_input_size=4, edge_output_size=6,
    global_input_size=2, global_output_size=4,
)
net = GraphNet(**full)
out = net.eval_tensor_dict(td_glob.copy())
assert out["nodes"].shape[-1] == 6
assert out["edges"].shape[-1] == 6
assert out["global_attr"].shape[-1] == 4

td_ng = {k: v for k, v in td_glob.items() if "global" not in k}
mpnn = make_mpnn_graphnet_noglobal_functions(
    units=8, node_or_core_input_size=3, node_or_core_output_size=6,
    edge_input_size=4, edge_output_size=6,
)
net_mpnn = GraphNet(**mpnn)
out = net_mpnn.eval_tensor_dict(td_ng.copy())
assert out["nodes"].shape[-1] == 6
assert "global_attr" not in out

n0 = Node(tf.ones((1, 3))); n1 = Node(tf.ones((1, 3)) * 2); n2 = Node(tf.ones((1, 3)) * 3)
g = Graph(
    [n0, n1, n2],
    [Edge(tf.ones((1, 4)), n0, n1), Edge(tf.ones((1, 4)) * 2, n1, n2)],
)

# Object-graph dispatch works for no-global blocks only:
out_g = net_mpnn.graph_eval(g)
assert out_g.nodes[0].get_state().shape[-1] == 6
assert out_g.edges[0].edge_tensor.shape[-1] == 6
assert len(out_g.edges) == 2
```

### Aggregation Modes And Message Widths

```python
import tensorflow as tf
from tf_gnns.graphnet_utils import GraphNet, make_mlp_graphnet_functions

td = {
    "nodes": tf.ones((3, 3), dtype=tf.float32),
    "edges": tf.ones((3, 4), dtype=tf.float32),
    "senders": tf.constant([0, 1, 2], dtype=tf.int32),
    "receivers": tf.constant([1, 2, 0], dtype=tf.int32),
    "n_nodes": tf.constant([3], dtype=tf.int32),
    "n_edges": tf.constant([3], dtype=tf.int32),
    "n_graphs": tf.constant(1, dtype=tf.int32),
    "global_attr": tf.ones((1, 2), dtype=tf.float32),
    "global_reps_for_nodes": tf.constant([0, 0, 0], dtype=tf.int32),
    "global_reps_for_edges": tf.constant([0, 0, 0], dtype=tf.int32),
}
multipliers = {"mean": 1, "sum": 1, "max": 1, "min": 1,
               "mean_max": 2, "mean_max_min": 3, "mean_max_min_sum": 4}
for agg, mult in multipliers.items():
    args = make_mlp_graphnet_functions(
        units=8, node_input_size=3, node_output_size=6,
        edge_input_size=4, edge_output_size=6,
        global_input_size=2, global_output_size=6,
        create_global_function=True, use_global_input=True,
        use_global_to_edge=True, use_global_to_node=True,
        aggregation_function=agg,
    )
    net = GraphNet(**args)
    msg_width = next(inp.shape[-1] for inp in net.node_function.inputs
                     if "edge_state_agg" in inp.name)
    assert msg_width == mult * 6
    out = net.eval_tensor_dict(td.copy())
    assert out["nodes"].shape[-1] == 6
    assert out["global_attr"].shape[-1] == 6

# Per-path aggregation overrides for the global block:
args = make_mlp_graphnet_functions(
    units=8, node_input_size=3, node_output_size=6,
    edge_input_size=4, edge_output_size=6,
    global_input_size=2, global_output_size=6,
    create_global_function=True, use_global_input=True,
    use_global_to_edge=True, use_global_to_node=True,
    aggregation_function="sum",
    node_to_global_aggr_fn="max", edge_to_global_aggr_fn="min",
)
net2 = GraphNet(**args)
out2 = net2.eval_tensor_dict(td.copy())
assert out2["nodes"].shape[-1] == 6
assert out2["global_attr"].shape[-1] == 6
```

### Serialization Roundtrip

```python
import os
import tempfile
import numpy as np
import tensorflow as tf
from tf_gnns.graphnet_utils import GraphNet, make_graph_indep_graphnet_functions

td = {
    "nodes": tf.ones((3, 3), dtype=tf.float32),
    "edges": tf.ones((3, 4), dtype=tf.float32),
    "senders": tf.constant([0, 1, 2], dtype=tf.int32),
    "receivers": tf.constant([1, 2, 0], dtype=tf.int32),
    "n_nodes": tf.constant([3], dtype=tf.int32),
    "n_edges": tf.constant([3], dtype=tf.int32),
    "n_graphs": tf.constant(1, dtype=tf.int32),
}
args = make_graph_indep_graphnet_functions(
    units=8, node_or_core_input_size=3, node_or_core_output_size=6,
    edge_input_size=4, edge_output_size=6,
    create_global_function=False, use_global_input=False,
)
net = GraphNet(**args)
expected = net.eval_tensor_dict(td.copy())

path = tempfile.mkdtemp()
net.save(path)
restored = GraphNet.make_from_path(path)
got = restored.eval_tensor_dict(td.copy())
assert np.allclose(expected["nodes"], got["nodes"], atol=1e-6)
assert len(net.weights) == len(restored.weights)

blank = GraphNet(node_function=None, edge_function=None)
blank.load(path)
assert blank.node_function is not None

# Boundary: global_function is not serialized; a reloaded block falls back to
# the input global width instead of its configured global output width.
args_w = make_graph_indep_graphnet_functions(
    units=8, node_or_core_input_size=3, node_or_core_output_size=6,
    edge_input_size=4, edge_output_size=6,
    global_input_size=2, global_output_size=4,
)
net_w = GraphNet(**args_w)
td_w = dict(td); td_w["global_attr"] = tf.ones((1, 2), dtype=tf.float32)
td_w["global_reps_for_nodes"] = tf.constant([0, 0, 0], dtype=tf.int32)
td_w["global_reps_for_edges"] = tf.constant([0, 0, 0], dtype=tf.int32)
assert net_w.eval_tensor_dict(td_w.copy())["global_attr"].shape[-1] == 4
path_w = tempfile.mkdtemp()
net_w.save(path_w)
restored_w = GraphNet.make_from_path(path_w)
assert restored_w.eval_tensor_dict(td_w.copy())["global_attr"].shape[-1] == 2
```

### Recurrent, Residual, And GNCellMLP

```python
import tensorflow as tf
from tf_gnns.models.graphnet import GNCellMLP, GraphNetMLP, GraphNetMPNN_MLP

td_ng = {
    "nodes": tf.ones((3, 3), dtype=tf.float32),
    "edges": tf.ones((3, 4), dtype=tf.float32),
    "senders": tf.constant([0, 1, 2], dtype=tf.int32),
    "receivers": tf.constant([1, 2, 0], dtype=tf.int32),
    "n_nodes": tf.constant([3], dtype=tf.int32),
    "n_edges": tf.constant([3], dtype=tf.int32),
    "n_graphs": tf.constant(1, dtype=tf.int32),
}

m_rec = GraphNetMPNN_MLP(units=8, core_steps=3, recurrent=True)
m_non = GraphNetMPNN_MLP(units=8, core_steps=3, recurrent=False)
# Call both first: `all_weights` is populated during build/lazily on first call.
# Output node width defaults to the input width (3) unless `node_output_size` is set.
m_rec(td_ng.copy())
m_non(td_ng.copy())
assert len(m_rec.all_weights) < len(m_non.all_weights)  # recurrent shares one core block
assert m_rec(td_ng.copy())["nodes"].shape[-1] == 3
assert m_non(td_ng.copy())["nodes"].shape[-1] == 3

m_nores = GraphNetMPNN_MLP(units=8, core_steps=2, residual=False)
assert m_nores(td_ng.copy())["nodes"].shape[-1] == 3

td_g = dict(td_ng)
td_g["global_attr"] = tf.ones((1, 2), dtype=tf.float32)
td_g["global_reps_for_nodes"] = tf.constant([0, 0, 0], dtype=tf.int32)
td_g["global_reps_for_edges"] = tf.constant([0, 0, 0], dtype=tf.int32)
m_g = GraphNetMLP(units=8, core_steps=2, recurrent=True)
assert m_g(td_g.copy())["nodes"].shape[-1] == 3

# GNCellMLP is a single one-step block; integer `gn_mlp_units` honors the
# requested sizes (a list would force the output to the last element instead).
cell = GNCellMLP(gn_mlp_units=8, core_size=6)
out = cell(td_g.copy())
assert out["nodes"].shape[-1] == 6
assert out["edges"].shape[-1] == 6
assert out["global_attr"].shape[-1] == 6
```
