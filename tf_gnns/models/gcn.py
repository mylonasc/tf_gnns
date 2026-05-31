"""Backend-agnostic sparse GCN layers and models.

This module provides:
- ``SparseGCNConv``: a single graph convolution layer over GraphTuple-like
  tensor dictionaries.
- ``SparseGCN``: a multi-layer stack for node-level prediction.

The implementation uses only ``keras.ops`` and backend facade ops from
``tf_gnns.backend_ops`` so it works across Keras backends.
"""

from __future__ import annotations

import keras

from .. import backend_ops


def _ensure_tensor(x, dtype=None):
    return backend_ops.convert_to_tensor(x, dtype=dtype)


class SparseGCNConv(keras.layers.Layer):
    """Single sparse GCN convolution layer.

    Inputs are expected to follow the graph tensor-dict format used by this
    repository (keys: ``nodes``, ``senders``, ``receivers``, and counts).
    """

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        add_self_loops=True,
        normalize=True,
        jit_compile=False,
        feature_dtype=None,
        index_dtype=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.jit_compile = bool(jit_compile)
        self.feature_dtype = feature_dtype
        self.index_dtype = index_dtype

    @property
    def _feature_dtype(self):
        if self.feature_dtype is not None:
            return self.feature_dtype
        return self.compute_dtype or keras.backend.floatx()

    @property
    def _index_dtype(self):
        if self.index_dtype is not None:
            return self.index_dtype
        if backend_ops.active_backend() == "torch":
            return "int64"
        return "int32"

    def build(self, input_shape):
        node_dim = int(input_shape["nodes"][-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(node_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            dtype=self._feature_dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
                dtype=self._feature_dtype,
            )
        else:
            self.bias = None

        super().build(input_shape)

    def _build_indices_and_weights(self, d):
        senders = keras.ops.cast(_ensure_tensor(d["senders"]), self._index_dtype)
        receivers = keras.ops.cast(_ensure_tensor(d["receivers"]), self._index_dtype)

        # Prefer static shape to avoid symbolic/meta scalar issues in some
        # backends (notably Keras torch during tracing/build).
        static_num_nodes = getattr(d["nodes"], "shape", [None])[0]
        if static_num_nodes is not None:
            num_nodes = int(static_num_nodes)
        else:
            num_nodes = keras.ops.cast(keras.ops.shape(d["nodes"])[0], self._index_dtype)

        if "edge_weights" in d and d["edge_weights"] is not None:
            weights = _ensure_tensor(d["edge_weights"], dtype=self._feature_dtype)
            weights = keras.ops.reshape(weights, (-1,))
        else:
            weights = keras.ops.ones_like(keras.ops.cast(senders, self._feature_dtype))

        row = receivers
        col = senders

        if self.add_self_loops:
            diag = keras.ops.arange(num_nodes, dtype=self._index_dtype)
            row = keras.ops.concatenate([row, diag], axis=0)
            col = keras.ops.concatenate([col, diag], axis=0)
            loop_w = keras.ops.ones_like(keras.ops.cast(diag, self._feature_dtype))
            weights = keras.ops.concatenate([weights, loop_w], axis=0)

        if self.normalize:
            deg = keras.ops.segment_sum(weights, row, num_nodes)
            deg = keras.ops.maximum(deg, keras.ops.ones_like(deg) * 1e-12)
            inv_sqrt_deg = keras.ops.rsqrt(deg)
            norm_w = (
                weights
                * keras.ops.take(inv_sqrt_deg, row, axis=0)
                * keras.ops.take(inv_sqrt_deg, col, axis=0)
            )
        else:
            norm_w = weights

        return norm_w, num_nodes, row, col

    def _segment_fallback(self, nodes, row, col, norm_w, num_nodes):
        gathered = keras.ops.take(nodes, col, axis=0)
        scaled = gathered * keras.ops.expand_dims(norm_w, axis=-1)
        return keras.ops.segment_sum(scaled, row, num_nodes)

    def call(self, d):
        nodes = _ensure_tensor(d["nodes"], dtype=self._feature_dtype)
        norm_w, num_nodes, row, col = self._build_indices_and_weights(d)
        agg = self._segment_fallback(nodes, row, col, norm_w, num_nodes)

        out_nodes = keras.ops.matmul(agg, self.kernel)
        if self.bias is not None:
            out_nodes = out_nodes + self.bias
        if self.activation is not None:
            out_nodes = self.activation(out_nodes)

        out = d.copy()
        out["nodes"] = out_nodes
        return out


class SparseGCN(keras.layers.Layer):
    """Stack of sparse GCN layers for node features."""

    def __init__(
        self,
        hidden_units,
        output_units=None,
        activation="relu",
        dropout_rate=0.0,
        add_self_loops=True,
        normalize=True,
        jit_compile=False,
        feature_dtype=None,
        index_dtype=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        if not hidden_units:
            raise ValueError("hidden_units must contain at least one layer width")

        self.dropout = keras.layers.Dropout(dropout_rate)
        self.feature_dtype = feature_dtype
        self.index_dtype = index_dtype
        widths = list(hidden_units)
        if output_units is not None:
            widths.append(output_units)

        self.convs = []
        for i, width in enumerate(widths):
            is_last = i == len(widths) - 1
            conv_activation = None if is_last else activation
            self.convs.append(
                SparseGCNConv(
                    units=width,
                    activation=conv_activation,
                    add_self_loops=add_self_loops,
                    normalize=normalize,
                    jit_compile=jit_compile,
                    feature_dtype=feature_dtype,
                    index_dtype=index_dtype,
                    name=f"sparse_gcn_conv_{i}",
                )
            )

    def call(self, d, training=False):
        out = d
        for i, conv in enumerate(self.convs):
            out = conv(out)
            if i < len(self.convs) - 1:
                out_nodes = self.dropout(out["nodes"], training=training)
                out = out.copy()
                out["nodes"] = out_nodes
        return out
