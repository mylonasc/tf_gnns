# Backends

Read this when choosing TensorFlow, Torch, or JAX Keras backends.

## How It Works

tf_gnns uses Keras 3 and a small `tf_gnns.backend_ops` facade for backend-portable tensor operations. The backend is selected at the Keras level before importing tf_gnns model layers. TensorFlow is the primary tested backend; Torch and JAX are supported for portability smoke tests.

Application-level compilation uses `@tf.function` wrapping the model call, not internal changes. All tf_gnns layers are standard Keras `Layer` subclasses and work with any Keras backend.

**Tensor dictionary format** (plain `dict` of tensors):

| Key | Shape | Description |
|---|---|---|
| `nodes` | [N, D_node] | Node features |
| `edges` | [E, D_edge] | Edge features |
| `senders` | [E] | Sender node indices |
| `receivers` | [E] | Receiver node indices |
| `n_nodes` | [K] | Nodes per graph (K graphs) |
| `n_edges` | [K] | Edges per graph |
| `n_graphs` | scalar | Number of graphs |
| `global_attr` | [K, D_glob] | Global features (optional) |
| `global_reps_for_nodes` | [N] | Graph index per node (optional) |
| `global_reps_for_edges` | [E] | Graph index per edge (optional) |

## Decision Guide

- **Standard use, full feature support?** → TensorFlow backend
- **Eager debugging, tight loop?** → TensorFlow (eager mode without `@tf.function`)
- **PyTorch ecosystem integration?** → Torch backend (limited tf_gnns feature coverage)
- **XLA compilation?** → JAX backend or TF `@tf.function(jit_compile=True)`

## Parameter Key Facts

| Parameter | Effect | Gotcha | Related |
|---|---|---|---|
| `@tf.function` | Wraps model call for tracing/compilation | Call the wrapper **twice** (1st traces, 2nd runs compiled) | `jit_compile=True` may fail on GPU |
| `training` | Passed to Keras layers | Must be explicit; controls dropout/batchnorm | Same in GCN layers |
| `global_attr` | Optional global features | Requires `global_reps_for_nodes` and `global_reps_for_edges` when present | — |
| `GraphNetMLP` | Portable model class | Works with any Keras backend | `GraphNetMPNN_MLP` for no-global |

## Boundaries & Gotchas

1. **`@tf.function(jit_compile=True)` may fail on GPU** with certain ops (e.g., `Rsqrt` XLA JIT). Use `jit_compile=False` or eager mode as fallback.
2. **Backend must be set before importing Keras-heavy modules.** Changing it after import has no effect.
3. **`tf.concat` requires `axis`.** `tf.concat([a, b])` raises TypeError in TF 2.21.
4. **Torch/JAX paths are for portability.** Full tf_gnns feature coverage is only tested on TensorFlow.

## Minimal Skeleton

```python
import tensorflow as tf
from tf_gnns.models.graphnet import GraphNetMPNN_MLP

# No-global MPNN path: `GraphNetMPNN_MLP` builds with create_global_function=False.
# (Use `GraphNetMLP` when the tensor dict includes `global_attr` + repetition keys.)
td = {
    "nodes": tf.ones((4, 3), dtype=tf.float32),
    "edges": tf.ones((4, 2), dtype=tf.float32),
    "senders": tf.constant([0, 1, 2, 3], dtype=tf.int32),
    "receivers": tf.constant([1, 2, 3, 0], dtype=tf.int32),
    "n_nodes": tf.constant([4], dtype=tf.int32),
    "n_edges": tf.constant([4], dtype=tf.int32),
    "n_graphs": tf.constant(1, dtype=tf.int32),
}
model = GraphNetMPNN_MLP(units=12, core_steps=2, node_output_size=4, edge_output_size=3)
out = model(td)
assert out["nodes"].shape == (4, 4)
assert "global_attr" not in out
```

## Rules

- `tf_gnns` uses Keras 3 and a small `tf_gnns.backend_ops` facade for backend-portable tensor operations.
- Choose a Keras backend before importing Keras-heavy modules.
- TensorFlow is the broadest tested backend for the full package and examples.
- Torch and JAX paths are intended for backend portability smoke tests and compatible Keras operations.
- TensorFlow graph compilation should be applied at the application level with `tf.function`, not by changing `tf_gnns` internals.

## Output Contract

- The tensor dictionary input keys and output keys are identical to the other topics: `nodes`/`edges` per-graph, plus `senders`/`receivers`/`n_nodes`/`n_edges`/`n_graphs`, and `global_attr` + `global_reps_for_nodes`/`global_reps_for_edges` when a global block is used.
- `GraphNetMPNN_MLP` is the no-global layer: with no `global_attr` in the input, its output has no `global_attr` key and node/edge widths follow `node_output_size`/`edge_output_size`.
- `GraphNetMLP` always builds a global block: it requires the full global-keyed dict and fails without it; its output `global_attr` width is `global_output_size`.
- Compiling with `@tf.function` around the wrapper, the output shapes equal the eager shapes; shape checking may be deferred until after one tracing call.

## API Signatures and Compilation Rules

- Use `@tf.function` **around the model wrapper**, not on object methods; call the wrapper **twice** so TF can build/trace.
- When compilation doesn't yield a clean tracing path, `model(td)` in eager mode still returns the same shapes.
- All `tf_gnns` layers are standard Keras `keras.layers.Layer` subclasses; pass `training=True`/`training=False` as usual.
- A tf_gnns tensor dictionary is a plain `dict` of `tf.Tensor`s with the keys shown above; add `global_attr` plus `global_reps_for_nodes`/`global_reps_for_edges` before layers with globals.
- `GraphTuple` as in graphs.md; `GraphTuple.assign_global(global_attr, check_shape=False)` adds globals to an existing batch.
- `GraphNetMLP/GraphNetMPNN_MLP` signatures as in graphnets.md; `GraphNet.eval_tensor_dict(td)` / `graph_tuple_eval(gt)` are the compiled/eager entry points.

## Example Gallery

### Backend Check Example

```python
from tf_gnns import backend_ops

print(backend_ops.active_backend())
```

### TensorFlow `tf.function` + GraphNetMLP Worked Example

```python
import tensorflow as tf
from tf_gnns.models.graphnet import GraphNetMLP

# Tensor dictionary for one graph with 4 nodes, 4 edges, global_attr and
# global repetition vectors (keys expected by tf_gnns Layers).
td = {
    "nodes": tf.ones((4, 3), dtype=tf.float32),
    "edges": tf.ones((4, 2), dtype=tf.float32),
    "senders": tf.constant([0, 1, 2, 3], dtype=tf.int32),
    "receivers": tf.constant([1, 2, 3, 0], dtype=tf.int32),
    "n_nodes": tf.constant([4], dtype=tf.int32),
    "n_edges": tf.constant([4], dtype=tf.int32),
    "n_graphs": tf.constant(1, dtype=tf.int32),
    "global_attr": tf.ones((1, 2), dtype=tf.float32),
    "global_reps_for_nodes": tf.constant([0, 0, 0, 0], dtype=tf.int32),
    "global_reps_for_edges": tf.constant([0, 0, 0, 0], dtype=tf.int32),
}

model = GraphNetMLP(
    units=12,
    core_steps=2,
    node_output_size=4,
    edge_output_size=3,
    global_output_size=2,
)


@tf.function
def forward(td_):
    return model(td_)


out_first = forward(td)   # 1st call may trace/build
out_second = forward(td)  # call compiled function twice
out_eager = model(td)     # plain eager call also works if tracing is messy

assert out_second["nodes"].shape == (4, 4), out_second["nodes"].shape
assert out_second["edges"].shape == (4, 3), out_second["edges"].shape
assert out_second["global_attr"].shape == (1, 2), out_second["global_attr"].shape

loss = tf.reduce_mean(
    tf.concat(
        values=[
            tf.reshape(out_second["nodes"], [-1]),
            tf.reshape(out_second["edges"], [-1]),
            tf.reshape(out_second["global_attr"], [-1]),
        ],
        axis=0,
    )
)
assert bool(tf.math.is_finite(loss).numpy()), "loss is not finite"
```
