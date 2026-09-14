# Sparse GCN

Read this when using `SparseGCNConv`, `SparseGCN`, or `GCNv2` on tensor dictionaries.

## Rules

- Import GCN layers from `tf_gnns.models.gcn`; `GCNv2` is not exported at package root.
- Tensor dictionaries must include `nodes`, `senders`, and `receivers`; bookkeeping keys are passed through.
- Optional `edge_weights` is a scalar vector of length `E` (`float32` unless `feature_dtype` overrides it).
- `SparseGCNConv(units, activation, add_self_loops, normalize, feature_dtype, index_dtype)` supports explicit node-feature and index dtypes via `feature_dtype` and `index_dtype`.
- `SparseGCNConv` returns a tensor dictionary with updated `nodes`.
- `SparseGCN` stacks multiple `SparseGCNConv` layers and applies dropout between hidden layers.
- `GCNv2(num_layers=...)` with `num_layers < 1` raises `ValueError("num_layers must be >= 1")`.
- `GCNv2(batchnorm=True, layernorm=True)` raises `ValueError` ("supports either batchnorm or layernorm, not both").
- Do not set `batchnorm=True` and `layernorm=True` together on any GCN layer.

## SparseGCN Example

```python
import tensorflow as tf
from tf_gnns.models.gcn import SparseGCN, SparseGCNConv

td = {
    "nodes": tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=tf.float32),
    "senders": tf.constant([0, 1], dtype=tf.int32),
    "receivers": tf.constant([1, 2], dtype=tf.int32),
    "n_nodes": tf.constant([3], dtype=tf.int32),
    "n_edges": tf.constant([2], dtype=tf.int32),
    "n_graphs": tf.constant(1, dtype=tf.int32),
}
model = SparseGCN(hidden_units=[4], output_units=2, activation="relu")
out = model(td, training=False)
assert out["nodes"].shape == (3, 2)

conv = SparseGCNConv(units=4, activation="relu")
conv_out = conv(td, training=False)
assert conv_out["nodes"].shape == (3, 4)
```

## Edge Weights And Dtype Example

```python
import tensorflow as tf
from tf_gnns.models.gcn import SparseGCN, SparseGCNConv

td = {
    "nodes": tf.ones((4, 3), dtype=tf.float64),
    "senders": tf.constant([0, 1, 2, 3], dtype=tf.int64),
    "receivers": tf.constant([1, 2, 3, 0], dtype=tf.int64),
    "n_nodes": tf.constant([4], dtype=tf.int64),
    "n_edges": tf.constant([4], dtype=tf.int64),
    "n_graphs": tf.constant(1, dtype=tf.int64),
    "edge_weights": tf.constant([1.0, 1.0, 1.0, 1.0], dtype=tf.float64),
}
conv = SparseGCNConv(
    units=5, activation=None, add_self_loops=True, normalize=True,
    feature_dtype="float64", index_dtype="int64",
)
out = conv(td, training=False)
assert out["nodes"].shape == (4, 5)
assert out["nodes"].dtype == tf.float64

stack = SparseGCN(hidden_units=[6, 6], output_units=3, dropout_rate=0.0,
                  residual=True, residual_projection=True)
out2 = stack(td, training=False)
assert out2["nodes"].shape == (4, 3)
```

## API Signatures (authoritative, no need to read source)

- `SparseGCNConv(units, activation=None, add_self_loops=True, normalize=True, batchnorm=True, layernorm=False, feature_dtype=None, index_dtype=None)` — call with `training=False when outside a Keras fit loop`.
- `SparseGCN(hidden_units, output_units=None, activation="relu", dropout_rate=0.0, add_self_loops=True, normalize=True, batchnorm=True, layernorm=False, jit_compile=False, residual=False, residual_projection=False, feature_dtype=None, index_dtype=None, **kwargs)` — accepts `hidden_units` as an int (single layer) or a list of widths; `output_units` appends a final dense head.
- `GCNv2(hidden_units, output_units, num_layers=3, add_self_loops=True, normalize=True, residual=True, residual_projection=True, batchnorm=True, layernorm=False, input_dropout_rate=0.0, dropout_rate=0.0, jit_compile=False, feature_dtype=None, index_dtype=None, use_shortcut=True, use_bias=True)`.

## Output Contract

- `SparseGCNConv` and `SparseGCN` return a tensor dictionary with updated `nodes` and all bookkeeping keys (`senders`, `receivers`, `n_nodes`, ...) unchanged; pass through `edge_weights` if present in the input.
- Node widths: `SparseGCNConv(units)` -> `nodes` width `units`; `SparseGCN(hidden_units=[h, ...], output_units=o)` -> `nodes` width `o` when `output_units` is set.
- `GCNv2` returns a tensor dictionary with `nodes` of width `output_units`.
- Dictation of dtypes: `feature_dtype` controls node/weight dtypes, `index_dtype` controls `senders`/`receivers`/`n_nodes`; `edge_weights` follows `feature_dtype`.

## GCNv2 Example

```python
import tensorflow as tf
from tf_gnns.models.gcn import GCNv2

td = {
    "nodes": tf.ones((4, 3), dtype=tf.float32),
    "senders": tf.constant([0, 1, 2], dtype=tf.int32),
    "receivers": tf.constant([1, 2, 3], dtype=tf.int32),
}
model = GCNv2(hidden_units=8, output_units=5, num_layers=2)
out = model(td, training=False)
assert out["nodes"].shape == (4, 5)
```
