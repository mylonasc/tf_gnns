"""Sparse high-performance GCN layers and models.

This module provides:
- ``SparseGCNConv``: a single graph convolution layer over GraphTuple-like
  tensor dictionaries.
- ``SparseGCN``: a multi-layer stack for node-level prediction.

The fast TensorFlow path uses ``tf.sparse.sparse_dense_matmul`` and can be
JIT-compiled with ``tf.function(jit_compile=True)``.
"""

from __future__ import annotations

import keras
import tensorflow as tf

from .. import backend_ops


def _ensure_tensor(x, dtype=None):
    return backend_ops.convert_to_tensor(x, dtype=dtype)


def _as_int32(x):
    return keras.ops.cast(_ensure_tensor(x), "int32")


def _as_float32(x):
    return keras.ops.cast(_ensure_tensor(x), "float32")


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
        backend_fallback_segment=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.jit_compile = bool(jit_compile)
        self.backend_fallback_segment = bool(backend_fallback_segment)
        self._spmm = None

    def build(self, input_shape):
        node_dim = int(input_shape["nodes"][-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(node_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None

        def _spmm_fn(sp_adj, dense_nodes):
            return tf.sparse.sparse_dense_matmul(sp_adj, dense_nodes)

        self._spmm = tf.function(_spmm_fn, jit_compile=self.jit_compile)
        super().build(input_shape)

    def _build_indices_and_weights(self, d):
        senders = _as_int32(d["senders"])
        receivers = _as_int32(d["receivers"])
        num_nodes = _as_int32(keras.ops.shape(d["nodes"]))[0]

        if "edge_weights" in d and d["edge_weights"] is not None:
            weights = _as_float32(d["edge_weights"])
            weights = keras.ops.reshape(weights, (-1,))
        else:
            weights = keras.ops.ones_like(keras.ops.cast(senders, "float32"))

        row = receivers
        col = senders

        if self.add_self_loops:
            diag = keras.ops.arange(num_nodes, dtype="int32")
            row = keras.ops.concatenate([row, diag], axis=0)
            col = keras.ops.concatenate([col, diag], axis=0)
            loop_w = keras.ops.ones_like(keras.ops.cast(diag, "float32"))
            weights = keras.ops.concatenate([weights, loop_w], axis=0)

        if self.normalize:
            deg = backend_ops.segment_sum(weights, row, num_nodes)
            deg = keras.ops.maximum(deg, keras.ops.ones_like(deg) * 1e-12)
            inv_sqrt_deg = keras.ops.rsqrt(deg)
            norm_w = (
                weights
                * keras.ops.take(inv_sqrt_deg, row, axis=0)
                * keras.ops.take(inv_sqrt_deg, col, axis=0)
            )
        else:
            norm_w = weights

        indices = keras.ops.stack([row, col], axis=1)
        return indices, norm_w, num_nodes, row, col

    def _segment_fallback(self, nodes, row, col, norm_w, num_nodes):
        gathered = keras.ops.take(nodes, col, axis=0)
        scaled = gathered * keras.ops.expand_dims(norm_w, axis=-1)
        return backend_ops.segment_sum(scaled, row, num_nodes)

    def call(self, d):
        nodes = _ensure_tensor(d["nodes"], dtype="float32")
        indices, norm_w, num_nodes, row, col = self._build_indices_and_weights(d)

        if backend_ops.active_backend() == "tensorflow" and not self.backend_fallback_segment:
            sp_adj = tf.sparse.SparseTensor(
                indices=tf.cast(indices, tf.int64),
                values=tf.cast(norm_w, tf.float32),
                dense_shape=tf.cast(tf.stack([num_nodes, num_nodes]), tf.int64),
            )
            sp_adj = tf.sparse.reorder(sp_adj)
            agg = self._spmm(sp_adj, tf.cast(nodes, tf.float32))
        else:
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        if not hidden_units:
            raise ValueError("hidden_units must contain at least one layer width")

        self.dropout = keras.layers.Dropout(dropout_rate)
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
