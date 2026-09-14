# Backends

Read this when choosing TensorFlow, Torch, or JAX Keras backends.

## Rules

- `tf_gnns` uses Keras 3 and a small `tf_gnns.backend_ops` facade for backend-portable tensor operations.
- Choose a Keras backend before importing Keras-heavy modules.
- TensorFlow is the broadest tested backend for the full package and examples.
- Torch and JAX paths are intended for backend portability smoke tests and compatible Keras operations.
- TensorFlow graph compilation should be applied at the application level with `tf.function`, not by changing `tf_gnns` internals.

## Backend Check Example

```python
from tf_gnns import backend_ops

print(backend_ops.active_backend())
```

## TensorFlow `tf.function` + GraphNetMLP Worked Example

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

## API Signatures and Compilation Rules

- Use `@tf.function` **around the model wrapper**, not on object methods; call the wrapper **twice** so TF can build/trace.
- When compilation doesn't yield a clean tracing path, `model(td)` in eager mode still returns the same shapes.
- All `tf_gnns` layers are standard Keras `keras.layers.Layer` subclasses; pass `training=True`/`training=False` as usual.
- A tf_gnns tensor dictionary is a plain `dict` of `tf.Tensor`s with the keys shown above; add `global_attr` plus `global_reps_for_nodes`/`global_reps_for_edges` before layers with globals.
- `GraphTuple` as in graphs.md; `GraphTuple.assign_global(global_attr, check_shape=False)` adds globals to an existing batch.
- `GraphNetMLP/GraphNetMPNN_MLP` signatures as in graphnets.md; `GraphNet.eval_tensor_dict(td)` / `graph_tuple_eval(gt)` are the compiled/eager entry points.
