import numpy as np
import tensorflow as tf

from tf_gnns.models.gcn import SparseGCN, SparseGCNConv
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


def test_sparse_gcn_conv_segment_fallback_matches_spmm_close():
    td = _make_td(edge_weights=[1.0, 0.75, 1.25])
    spmm_layer = SparseGCNConv(
        units=3,
        activation=None,
        jit_compile=False,
        backend_fallback_segment=False,
        add_self_loops=True,
        normalize=True,
    )
    seg_layer = SparseGCNConv(
        units=3,
        activation=None,
        jit_compile=False,
        backend_fallback_segment=True,
        add_self_loops=True,
        normalize=True,
    )

    _ = spmm_layer(td)
    _ = seg_layer(td)
    seg_layer.set_weights(spmm_layer.get_weights())

    out_spmm = _to_numpy(spmm_layer(td)["nodes"])
    out_seg = _to_numpy(seg_layer(td)["nodes"])
    np.testing.assert_allclose(out_spmm, out_seg, rtol=1e-5, atol=1e-5)


def test_sparse_gcn_stack_output_shape():
    td = _make_td()
    model = SparseGCN(hidden_units=[8, 8], output_units=2, dropout_rate=0.0)
    out = model(td, training=False)
    assert out["nodes"].shape == (3, 2)
