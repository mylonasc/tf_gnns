# Sparse GCN

Read this when using `SparseGCNConv`, `SparseGCN`, or `GCNv2` on tensor dictionaries.

## How It Works

Sparse GCN layers process graph-structured data using a sparse adjacency matrix representation. The flow is: **node features [N, D] × sparse adjacency → aggregated neighbor features → dense transformation → output nodes [N, units]**.

`SparseGCNConv` is the single-layer primitive: it normalizes the adjacency, aggregates neighbor features, and applies a dense projection. `SparseGCN` stacks multiple `SparseGCNConv` layers with optional dropout, residual connections, and batch/layer norm. `GCNv2` is the high-level recipe wrapper that configures the full stack (hidden layers + output head) matching the tunedGNN OGB-Arxiv pattern.

All GCN layers take `training=True/False`: dropout and batchnorm engage only when `training=True`. The `normalize` parameter controls adjacency normalization (default `True` for training mode; set `False` for inference-only to avoid the GPU rsqrt path).

**Shape table** (4 nodes, 3-dim features, no edge weights):

| Layer | Input `nodes` | Output `nodes` | Key params |
|---|---|---|---|
| `SparseGCNConv(units=5)` | [4, 3] | [4, 5] | single layer |
| `SparseGCN(hidden_units=[6,6], output_units=3)` | [4, 3] | [4, 3] | 2 hidden + output head |
| `GCNv2(hidden_units=8, output_units=4, num_layers=3)` | [4, 3] | [4, 4] | 3 hidden + output head |

## Decision Guide

- **Single graph convolution?** → `SparseGCNConv(units=...)`
- **Multi-layer with residual/dropout?** → `SparseGCN(hidden_units=[...], output_units=...)`
- **Full OGB-Arxiv recipe (hidden stack + output head)?** → `GCNv2(hidden_units=..., output_units=..., num_layers=...)`

## Parameter Key Facts

| Parameter | Effect | Gotcha | Related |
|---|---|---|---|
| `hidden_units` | Width of hidden layers (int or list) | Int = single hidden layer; list = stack of layers | `num_layers` in GCNv2 |
| `output_units` | Width of final output head | When set, appends a final dense layer | — |
| `training` | Enables dropout/batchnorm | Must be passed explicitly to `call()` | Both paths return same shapes |
| `dropout_rate` | Fraction of neurons dropped | Only active when `training=True` | `input_dropout_rate` (GCNv2) |
| `residual` | Add input to output (skip connection) | Requires `residual_projection=True` when input dim ≠ output dim | — |
| `batchnorm` / `layernorm` | Normalization after each layer | **Mutually exclusive** — setting both raises `ValueError` | — |
| `normalize` | Adjacency normalization | Default `True`; `False` avoids GPU rsqrt crash | — |
| `feature_dtype` | Node/weight dtype | Controls float32 vs float64 | `index_dtype` for indices |
| `edge_weights` | Optional scalar weights per edge | Length must match number of edges | — |

## Boundaries & Gotchas

1. **GPU XLA rsqrt crash.** On certain GPU driver versions (595.x), `normalize=True` triggers an XLA JIT failure on `Rsqrt`. Workaround: set `normalize=False` for inference-only, or set `XLA_FLAGS=--xla_gpu_cuda_data_dir=...`.
2. **`batchnorm=True` + `layernorm=True` raises ValueError.** GCNv2 and SparseGCN do not support both.
3. **`GCNv2(num_layers=0)` raises ValueError.** Must be ≥ 1.
4. **`training` must be passed explicitly.** Forgetting `training=True` in a fit loop silently skips dropout/batchnorm.
5. **`tf.is_finite` → `tf.math.is_finite`** in TF 2.21.

## Minimal Skeleton

```python
import tensorflow as tf
from tf_gnns.models.gcn import SparseGCNConv

td = {
    "nodes": tf.ones((4, 3), dtype=tf.float32),
    "senders": tf.constant([0, 1, 2], dtype=tf.int32),
    "receivers": tf.constant([1, 2, 3], dtype=tf.int32),
}
out = SparseGCNConv(units=5, activation=None)(td, training=False)
assert out["nodes"].shape == (4, 5)
```

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

## API Signatures (authoritative, no need to read source)

- `SparseGCNConv(units, activation=None, add_self_loops=True, normalize=True, batchnorm=True, layernorm=False, feature_dtype=None, index_dtype=None)` — call with `training=False when outside a Keras fit loop`.
- `SparseGCN(hidden_units, output_units=None, activation="relu", dropout_rate=0.0, add_self_loops=True, normalize=True, batchnorm=True, layernorm=False, jit_compile=False, residual=False, residual_projection=False, feature_dtype=None, index_dtype=None, **kwargs)` — accepts `hidden_units` as an int (single layer) or a list of widths; `output_units` appends a final dense head.
- `GCNv2(hidden_units, output_units, num_layers=3, add_self_loops=True, normalize=True, residual=True, residual_projection=True, batchnorm=True, layernorm=False, input_dropout_rate=0.0, dropout_rate=0.0, jit_compile=False, feature_dtype=None, index_dtype=None, use_shortcut=True, use_bias=True)`.
- All GCN layers take a `training=True/False` argument on their `call`; dropout and batchnorm engage only when `training=True`.

## Output Contract

- `SparseGCNConv` and `SparseGCN` return a tensor dictionary with updated `nodes` and all bookkeeping keys (`senders`, `receivers`, `n_nodes`, ...) unchanged; pass through `edge_weights` if present in the input.
- Node widths: `SparseGCNConv(units)` -> `nodes` width `units`; `SparseGCN(hidden_units=[h, ...], output_units=o)` -> `nodes` width `o` when `output_units` is set.
- `GCNv2` returns a tensor dictionary with `nodes` of width `output_units`.
- Dictation of dtypes: `feature_dtype` controls node/weight dtypes, `index_dtype` controls `senders`/`receivers`/`n_nodes`; `edge_weights` follows `feature_dtype`.

## Example Gallery

### SparseGCN Example

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

### Edge Weights And Dtype Example

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

### Training Mode (Dropout And BatchNorm)

```python
import tensorflow as tf
from tf_gnns.models.gcn import GCNv2, SparseGCN

td = {
    "nodes": tf.ones((4, 3), dtype=tf.float32),
    "senders": tf.constant([0, 1, 2, 3], dtype=tf.int32),
    "receivers": tf.constant([1, 2, 3, 0], dtype=tf.int32),
    "n_nodes": tf.constant([4], dtype=tf.int32),
    "n_edges": tf.constant([4], dtype=tf.int32),
    "n_graphs": tf.constant(1, dtype=tf.int32),
}
stack = SparseGCN(hidden_units=[6, 6], output_units=3, dropout_rate=0.3,
                  residual=True, residual_projection=True)
train_out = stack(td, training=True)
assert train_out["nodes"].shape == (4, 3)
assert tf.reduce_all(tf.math.is_finite(train_out["nodes"])).numpy()

model = GCNv2(hidden_units=8, output_units=4, num_layers=3,
              input_dropout_rate=0.2, dropout_rate=0.3,
              residual=True, residual_projection=True, batchnorm=True)
out = model(td, training=True)
assert out["nodes"].shape == (4, 4)
assert tf.reduce_all(tf.math.is_finite(out["nodes"])).numpy()
```

### GCNv2 Example

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
