import numpy as np
import pytest
import tensorflow as tf

from tf_gnns.models.gcn import GCNv2, SparseGCN, SparseGCNConv
from tf_gnns.tfgnns_datastructures import GraphTuple


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    return tf.keras.ops.convert_to_numpy(x)


def _make_td(edge_weights=None):
    nodes = tf.constant([[1.0, 2.0], [0.5, -1.0], [3.0, 1.0]], dtype=tf.float32)
    edges = tf.zeros((3, 1), dtype=tf.float32)
    senders = tf.constant([0, 1, 2], dtype=tf.int32)
    receivers = tf.constant([1, 2, 0], dtype=tf.int32)
    gt = GraphTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_nodes=[3],
        n_edges=[3],
        global_attr=None,
    )
    td = gt.to_tensor_dict()
    if edge_weights is not None:
        td["edge_weights"] = tf.constant(edge_weights, dtype=tf.float32)
    return td


def test_sparse_gcn_conv_preserves_graph_structure_and_updates_nodes():
    td = _make_td()
    layer = SparseGCNConv(units=4, activation="relu", jit_compile=False)
    out = layer(td)

    assert out["nodes"].shape == (3, 4)
    tf.debugging.assert_equal(out["senders"], td["senders"])
    tf.debugging.assert_equal(out["receivers"], td["receivers"])


def test_sparse_gcn_conv_repeated_layers_with_shared_weights_match():
    td = _make_td(edge_weights=[1.0, 0.75, 1.25])
    layer_a = SparseGCNConv(
        units=3,
        activation=None,
        jit_compile=False,
        add_self_loops=True,
        normalize=True,
    )
    layer_b = SparseGCNConv(
        units=3,
        activation=None,
        jit_compile=False,
        add_self_loops=True,
        normalize=True,
    )

    _ = layer_a(td)
    _ = layer_b(td)
    layer_b.set_weights(layer_a.get_weights())

    out_a = _to_numpy(layer_a(td)["nodes"])
    out_b = _to_numpy(layer_b(td)["nodes"])
    np.testing.assert_allclose(out_a, out_b, rtol=1e-5, atol=1e-5)


def test_sparse_gcn_stack_output_shape():
    td = _make_td()
    model = SparseGCN(hidden_units=[8, 8], output_units=2, dropout_rate=0.0)
    out = model(td, training=False)
    assert out["nodes"].shape == (3, 2)


def test_sparse_gcn_stack_supports_residual_hidden_layers():
    td = _make_td()
    model = SparseGCN(
        hidden_units=[8, 8],
        output_units=2,
        dropout_rate=0.0,
        residual=True,
    )
    out = model(td, training=False)
    assert out["nodes"].shape == (3, 2)


def test_sparse_gcn_residual_projection_handles_width_mismatch():
    td = _make_td()
    model = SparseGCN(
        hidden_units=[7, 5],
        output_units=2,
        dropout_rate=0.0,
        residual=True,
        residual_projection=True,
    )
    out = model(td, training=False)
    assert out["nodes"].shape == (3, 2)


def test_sparse_gcn_residual_without_projection_skips_mismatched_hidden_skip():
    td = _make_td()
    model = SparseGCN(
        hidden_units=[7, 5],
        output_units=2,
        dropout_rate=0.0,
        residual=True,
        residual_projection=False,
    )
    out = model(td, training=False)
    assert out["nodes"].shape == (3, 2)


def test_sparse_gcn_supports_layernorm_when_batchnorm_disabled():
    td = _make_td()
    model = SparseGCN(
        hidden_units=[8, 8],
        output_units=2,
        dropout_rate=0.0,
        batchnorm=False,
        layernorm=True,
    )
    out = model(td, training=False)
    assert out["nodes"].shape == (3, 2)


def test_sparse_gcn_raises_when_batchnorm_and_layernorm_both_enabled():
    with pytest.raises(ValueError):
        _ = SparseGCN(hidden_units=[8], batchnorm=True, layernorm=True)


def test_sparse_gcn_conv_raises_when_batchnorm_and_layernorm_both_enabled():
    with pytest.raises(ValueError):
        _ = SparseGCNConv(units=8, batchnorm=True, layernorm=True)


def test_sparse_gcn_supports_configured_feature_and_index_dtypes():
    td = _make_td(edge_weights=[1.0, 0.75, 1.25])
    td["nodes"] = tf.cast(td["nodes"], tf.float64)
    td["edge_weights"] = tf.cast(td["edge_weights"], tf.float64)
    td["senders"] = tf.cast(td["senders"], tf.int64)
    td["receivers"] = tf.cast(td["receivers"], tf.int64)

    layer = SparseGCNConv(
        units=4,
        activation=None,
        feature_dtype="float64",
        index_dtype="int64",
    )
    out = layer(td)

    assert "float64" in str(layer.kernel.dtype)
    assert "float64" in str(out["nodes"].dtype)


def test_gcnv2_stack_output_shape():
    td = _make_td()
    model = GCNv2(
        hidden_units=8,
        output_units=2,
        num_layers=3,
        input_dropout_rate=0.1,
        dropout_rate=0.2,
        batchnorm=True,
        layernorm=False,
    )
    out = model(td, training=False)
    assert out["nodes"].shape == (3, 2)


def test_gcnv2_raises_when_batchnorm_and_layernorm_both_enabled():
    with pytest.raises(ValueError):
        _ = GCNv2(
            hidden_units=8,
            output_units=2,
            num_layers=3,
            batchnorm=True,
            layernorm=True,
        )


def test_gcnv2_raises_on_invalid_num_layers():
    with pytest.raises(ValueError):
        _ = GCNv2(hidden_units=8, output_units=2, num_layers=0)


def test_gcnv2_residual_projection_handles_input_width_mismatch():
    td = _make_td()
    model = GCNv2(
        hidden_units=8,
        output_units=2,
        num_layers=3,
        residual=True,
        residual_projection=True,
    )
    out = model(td, training=False)
    assert out["nodes"].shape == (3, 2)


def test_gcnv2_residual_without_projection_skips_mismatched_first_skip():
    td = _make_td()
    model = GCNv2(
        hidden_units=8,
        output_units=2,
        num_layers=3,
        residual=True,
        residual_projection=False,
    )
    out = model(td, training=False)
    assert out["nodes"].shape == (3, 2)
