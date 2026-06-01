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
    r"""Single sparse GCN convolution layer.

    Inputs are expected to follow the graph tensor-dict format used by this
    repository (keys: ``nodes``, ``senders``, ``receivers``, and counts).

    This layer supports a paper-faithful update structure with a separate root
    transform, normalized neighborhood aggregation, and optional BatchNorm:

    .. math::

        h_v^{l} = \sigma\left(\mathrm{Norm}\left(
            h_v^{l-1}W_r^l + \sum_{u \in \mathcal{N}(v) \cup \{v\}}
            \frac{1}{\sqrt{\hat d_u \hat d_v}} h_u^{l-1}W^l
        \right)\right)

    where dropout is typically applied by the outer :class:`SparseGCN` stack.

    Reference:
        "Classic GNNs are Strong Baselines" (2024),
        https://arxiv.org/pdf/2406.08993
    """

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        add_self_loops=True,
        normalize=True,
        use_root_transform=True,
        batchnorm=True,
        layernorm=False,
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
        self.use_root_transform = bool(use_root_transform)
        self.batchnorm = bool(batchnorm)
        self.layernorm = bool(layernorm)
        if self.batchnorm and self.layernorm:
            raise ValueError(
                "SparseGCNConv supports either batchnorm or layernorm, not both."
            )
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
        self.root_kernel = None
        if self.use_root_transform:
            self.root_kernel = self.add_weight(
                name="root_kernel",
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

        self.bn = None
        if self.batchnorm:
            self.bn = keras.layers.BatchNormalization(
                momentum=0.9,
                epsilon=1e-5,
                dtype=self._feature_dtype,
            )
        self.ln = None
        if self.layernorm:
            self.ln = keras.layers.LayerNormalization(dtype=self._feature_dtype)

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

    def call(self, d, training=False):
        nodes = _ensure_tensor(d["nodes"], dtype=self._feature_dtype)
        norm_w, num_nodes, row, col = self._build_indices_and_weights(d)
        agg = self._segment_fallback(nodes, row, col, norm_w, num_nodes)

        out_nodes = keras.ops.matmul(agg, self.kernel)
        if self.root_kernel is not None:
            out_nodes = out_nodes + keras.ops.matmul(nodes, self.root_kernel)
        if self.bias is not None:
            out_nodes = out_nodes + self.bias
        if self.bn is not None:
            out_nodes = self.bn(out_nodes, training=training)
        if self.ln is not None:
            out_nodes = self.ln(out_nodes)
        if self.activation is not None:
            out_nodes = self.activation(out_nodes)

        out = d.copy()
        out["nodes"] = out_nodes
        return out


class SparseGCN(keras.layers.Layer):
    """Multi-layer sparse GCN stack for node-level representations.

    This layer composes multiple :class:`SparseGCNConv` blocks and applies
    optional dropout between hidden layers. By default, each convolution uses
    BatchNorm, matching the "Norm" term used in common strong-baseline GCN
    formulations. Inputs follow the graph tensor-dict structure used across
    ``tf_gnns``:

    - ``nodes``: node feature matrix with shape ``[N, F]``
    - ``senders``: source-node indices for edges, shape ``[E]``
    - ``receivers``: target-node indices for edges, shape ``[E]``
    - ``edge_weights`` (optional): scalar edge weights, shape ``[E]``
    - graph bookkeeping keys such as ``n_nodes``/``n_edges`` are passed through

    The stack performs repeated message passing in sparse form. Each
    ``SparseGCNConv`` can add self-loops and symmetric normalization
    ``D^{-1/2} A D^{-1/2}`` (configurable).

    Typical usage (2-layer GCN for node classification)::

        gcn = SparseGCN(
            hidden_units=[128],
            output_units=num_classes,
            activation="relu",
            dropout_rate=0.5,
            add_self_loops=True,
            normalize=True,
        )
        out_td = gcn(input_td, training=True)
        logits = out_td["nodes"]

    A deeper variant with custom dtypes::

        gcn = SparseGCN(
            hidden_units=[256, 256, 128],
            output_units=64,
            activation="relu",
            dropout_rate=0.2,
            feature_dtype="float32",
            index_dtype="int32",
        )

    Args:
        hidden_units: Width(s) of hidden GCN layers. Accepts an ``int`` for a
            single hidden layer or a non-empty list of ``int`` values.
        output_units: Optional final output width appended after hidden layers.
            For classification this is typically the number of classes.
        activation: Activation for hidden layers. Final layer uses ``None`` by
            default so downstream losses can consume logits directly.
        dropout_rate: Dropout probability applied to node features between
            convolution layers (not applied after the final layer).
        add_self_loops: Whether each ``SparseGCNConv`` injects identity edges.
        normalize: Whether each ``SparseGCNConv`` applies degree-based
            normalization to edge weights.
        batchnorm: Whether each ``SparseGCNConv`` applies BatchNormalization to
            node features after message/root aggregation and before activation.
            Enabled by default.
        layernorm: Whether each ``SparseGCNConv`` applies LayerNormalization to
            node features after message/root aggregation and before activation.
            Must not be enabled at the same time as ``batchnorm``.
        jit_compile: Flag forwarded to ``SparseGCNConv`` for backend-specific
            compiled execution behavior.
        residual: If ``True``, add residual node skip-connections between
            hidden layers when feature dimensions match. Residual links are not
            applied on the final layer.
        residual_projection: If ``True`` and ``residual`` is enabled, add a
            learnable linear projection on residual paths when hidden feature
            dimensions do not match. Projection is only used for hidden layers
            (not the final output layer).
        feature_dtype: Optional dtype override for feature tensors and weights.
        index_dtype: Optional dtype override for graph index tensors.
        **kwargs: Forwarded to ``keras.layers.Layer``.

    Returns:
        A tensor-dict with updated ``nodes`` and preserved graph structure keys.

    Raises:
        ValueError: If ``hidden_units`` is empty.
    """

    def __init__(
        self,
        hidden_units,
        output_units=None,
        activation="relu",
        dropout_rate=0.0,
        add_self_loops=True,
        normalize=True,
        batchnorm=True,
        layernorm=False,
        jit_compile=False,
        residual=False,
        residual_projection=False,
        feature_dtype=None,
        index_dtype=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if batchnorm and layernorm:
            raise ValueError(
                "SparseGCN supports either batchnorm or layernorm, not both."
            )
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        if not hidden_units:
            raise ValueError("hidden_units must contain at least one layer width")

        self.dropout = keras.layers.Dropout(dropout_rate)
        self.feature_dtype = feature_dtype
        self.index_dtype = index_dtype
        self.batchnorm = bool(batchnorm)
        self.layernorm = bool(layernorm)
        self.residual = bool(residual)
        self.residual_projection = bool(residual_projection)
        widths = list(hidden_units)
        if output_units is not None:
            widths.append(output_units)

        self.convs = []
        self.residual_projections = []
        for i, width in enumerate(widths):
            is_last = i == len(widths) - 1
            conv_activation = None if is_last else activation
            self.convs.append(
                SparseGCNConv(
                    units=width,
                    activation=conv_activation,
                    add_self_loops=add_self_loops,
                    normalize=normalize,
                    batchnorm=self.batchnorm,
                    layernorm=self.layernorm,
                    jit_compile=jit_compile,
                    feature_dtype=feature_dtype,
                    index_dtype=index_dtype,
                    name=f"sparse_gcn_conv_{i}",
                )
            )
            if self.residual_projection and not is_last:
                self.residual_projections.append(
                    keras.layers.Dense(
                        width,
                        use_bias=False,
                        dtype=feature_dtype,
                        name=f"sparse_gcn_residual_proj_{i}",
                    )
                )
            else:
                self.residual_projections.append(None)

    def call(self, d, training=False):
        """Apply stacked sparse graph convolutions.

        Args:
            d: Graph tensor-dict containing at least ``nodes``, ``senders``,
                and ``receivers``.
            training: Whether to run in training mode (controls dropout between
                hidden layers).

        Returns:
            Graph tensor-dict with transformed ``nodes``.
        """
        out = d
        for i, conv in enumerate(self.convs):
            prev_nodes = out["nodes"]
            out = conv(out, training=training)
            if self.residual and i < len(self.convs) - 1:
                skip = prev_nodes
                if prev_nodes.shape[-1] != out["nodes"].shape[-1]:
                    proj = self.residual_projections[i]
                    if proj is not None:
                        skip = proj(skip)
                    else:
                        skip = None
                if skip is not None:
                    out = out.copy()
                    out["nodes"] = out["nodes"] + skip
            if i < len(self.convs) - 1:
                out_nodes = self.dropout(out["nodes"], training=training)
                out = out.copy()
                out["nodes"] = out_nodes
        return out


class GCNv2(keras.layers.Layer):
    """High-level GCN stack matching tunedGNN's OGB-Arxiv recipe.

    This layer mirrors the common ``main-arxiv.py`` setup used in tunedGNN:

    - repeated hidden GCN layers (all with same hidden width),
    - each hidden layer uses the conv's internal root transform,
    - optional BatchNorm/LayerNorm, ReLU, and dropout per hidden layer,
    - optional input dropout before the first hidden layer,
    - final prediction via a dense linear head (not a graph conv).
    """

    def __init__(
        self,
        hidden_units,
        output_units,
        num_layers=3,
        add_self_loops=True,
        normalize=True,
        residual = True,
        residual_projection = True,
        batchnorm=True,
        layernorm=False,
        input_dropout_rate=0.0,
        dropout_rate=0.0,
        jit_compile=False,
        feature_dtype=None,
        index_dtype=None,
        use_shortcut=True, 
        use_bias = True,
        **kwargs,
    ):
        """Initializes a tunedGNN-style multi-layer GCN stack.

        This constructor configures a fixed-width hidden stack of
        :class:`SparseGCNConv` layers followed by a dense prediction head. The
        design is intended to mirror the effective forward path of tunedGNN's
        ``MPNNs`` model used on OGBN-Arxiv, where each hidden layer applies:

        1. graph convolution,
        2. residual/additive skip,
        3. normalization,
        4. ReLU,
        5. dropout.

        Args:
            hidden_units: Integer hidden width used for all graph-convolution
                layers.
            output_units: Integer output width for the final prediction head
                (typically number of classes for node classification).
            num_layers: Number of hidden graph-convolution layers. Must be
                greater than or equal to 1.
            add_self_loops: If ``True``, each convolution adds self-loops to
                edge indices internally. Set this to ``False`` when self-loops
                are already added during graph preprocessing to avoid
                double-counting.
            normalize: If ``True``, applies symmetric degree normalization in
                each convolution.
            residual: If ``True``, enables per-layer additive skip connections.
            residual_projection: If ``True`` (and ``residual`` is enabled),
                uses a learned linear projection on each residual branch.
                This most closely matches tunedGNN's
                ``GCNConv(...) + Linear(...)`` pattern.
            batchnorm: Enables BatchNormalization inside each hidden
                convolution layer.
            layernorm: Enables LayerNormalization inside each hidden
                convolution layer. Mutually exclusive with ``batchnorm``.
            input_dropout_rate: Dropout probability applied once on input node
                features before the first hidden layer.
            dropout_rate: Dropout probability applied after each hidden layer.
            jit_compile: Backend hint for compiled execution in lower-level
                ops where supported.
            feature_dtype: Optional floating-point dtype used for feature
                computations and trainable weights.
            index_dtype: Optional integer dtype used for graph indices.
            use_shortcut: If ``True``, accumulates intermediate hidden outputs
                into an extra shortcut path that is added before the final head.
            use_bias: If ``True``, enables bias terms in hidden convolutions.
            **kwargs: Additional keyword arguments forwarded to
                ``keras.layers.Layer``.

        Raises:
            ValueError: If both ``batchnorm`` and ``layernorm`` are enabled.
            ValueError: If ``num_layers < 1``.

        Notes:
            Batch normalization settings are highly sensitive for optimization
            on OGBN-Arxiv. In internal benchmarking, the BatchNorm hyper-
            parameters (especially ``momentum`` and ``epsilon``) significantly
            affect both convergence speed and best validation/test accuracy.
            Small mismatches from reference implementations can produce large
            performance gaps even when architecture and optimizer settings are
            otherwise aligned.
        """
        super().__init__(**kwargs)
        if batchnorm and layernorm:
            raise ValueError("GCNv2 supports either batchnorm or layernorm, not both.")
        if int(num_layers) < 1:
            raise ValueError("num_layers must be >= 1")
        self.use_shortcut = use_shortcut
        self.hidden_units = int(hidden_units)
        self.output_units = int(output_units)
        self.num_layers = int(num_layers)
        self.residual = bool(residual)
        self.residual_projection = bool(residual_projection)
        self.use_bias = use_bias
        self.input_dropout = keras.layers.Dropout(input_dropout_rate)
        self.dropout = keras.layers.Dropout(dropout_rate)

        self.convs = []
        self.residual_projections = []
        for i in range(self.num_layers):
            self.convs.append(
                SparseGCNConv(
                    units=self.hidden_units,
                    activation="relu",
                    add_self_loops=add_self_loops,
                    normalize=normalize,
                    use_root_transform=False,
                    batchnorm=batchnorm,
                    layernorm=layernorm,
                    jit_compile=jit_compile,
                    feature_dtype=feature_dtype,
                    index_dtype=index_dtype,
                    name=f"gcnv2_conv_{i}",
                    use_bias=self.use_bias
                )
            )
            if self.residual and self.residual_projection:
                self.residual_projections.append(
                    keras.layers.Dense(
                        self.hidden_units,
                        use_bias=True,
                        dtype=feature_dtype,
                        name=f"gcnv2_residual_proj_{i}",
                    )
                )
            else:
                self.residual_projections.append(None)
        self.head = keras.layers.Dense(
            self.output_units,
            activation=None,
            dtype=feature_dtype,
            name="gcnv2_head",
        )

    def call(self, d, training=False):
        out = d.copy()
        out["nodes"] = self.input_dropout(out["nodes"], training=training)
        if self.use_shortcut:
            shortcut = None
        for conv_idx, conv in enumerate(self.convs):
            prev_nodes = out["nodes"]
            out = conv(out, training=training)
            if self.residual:
                if self.residual_projection:
                    proj = self.residual_projections[conv_idx]
                    skip = proj(prev_nodes)
                else:
                    skip = prev_nodes
                    if prev_nodes.shape[-1] != out["nodes"].shape[-1]:
                        skip = None
                if skip is not None:
                    out = out.copy()
                    out["nodes"] = out["nodes"] + skip
            out_nodes = self.dropout(out["nodes"], training=training)
            # make shortcut connections of all layers to 
            # the end layer (skip the end layer)
            if self.use_shortcut and conv_idx < (len(self.convs)-1):
                contribution = out_nodes / len(self.convs)
                if shortcut is None:
                    shortcut = contribution
                else:
                    shortcut = shortcut + contribution
            out = out.copy()
            out["nodes"] = out_nodes

        if self.use_shortcut and shortcut is not None:
            out['nodes'] = out['nodes'] + shortcut
        out = out.copy()
        out["nodes"] = self.head(out["nodes"])
        return out
