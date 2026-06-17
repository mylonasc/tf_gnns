import numpy as np
import tensorflow as tf

from tf_gnns.lib.gt_ops import (
    _add_gt,
    _assign_add_tensor_dict,
    _concat_tensordicts,
    _slice_conc_tensordict,
    _zero_graph,
    _zero_graph_tf,
)


def _make_td(with_global=True):
    td = {
        "nodes": tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32),
        "edges": tf.constant([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=tf.float32),
        "senders": tf.constant([0, 0, 1], dtype=tf.int32),
        "receivers": tf.constant([1, 0, 1], dtype=tf.int32),
        "n_nodes": tf.constant([2], dtype=tf.int32),
        "n_edges": tf.constant([3], dtype=tf.int32),
        "n_graphs": tf.constant(1, dtype=tf.int32),
        "global_reps_for_edges": tf.constant([0, 0, 0], dtype=tf.int32),
        "global_reps_for_nodes": tf.constant([0, 0], dtype=tf.int32),
    }
    if with_global:
        td["global_attr"] = tf.constant([[11.0, 12.0]], dtype=tf.float32)
    return td


def test_assign_add_tensor_dict_adds_all_feature_fields():
    d1 = _make_td(with_global=True)
    d2 = _make_td(with_global=True)

    out = _assign_add_tensor_dict(d1.copy(), d2)

    tf.debugging.assert_equal(out["nodes"], d1["nodes"] + d2["nodes"])
    tf.debugging.assert_equal(out["edges"], d1["edges"] + d2["edges"])
    tf.debugging.assert_equal(out["global_attr"], d1["global_attr"] + d2["global_attr"])
    for k in [
        "senders",
        "receivers",
        "n_nodes",
        "n_edges",
        "n_graphs",
        "global_reps_for_edges",
        "global_reps_for_nodes",
    ]:
        tf.debugging.assert_equal(out[k], d1[k])


def test_add_gt_preserves_structure_and_adds_features():
    g1 = _make_td(with_global=True)
    g2 = _make_td(with_global=True)

    out = _add_gt(g1, g2)

    tf.debugging.assert_equal(out["nodes"], g1["nodes"] + g2["nodes"])
    tf.debugging.assert_equal(out["edges"], g1["edges"] + g2["edges"])
    tf.debugging.assert_equal(out["global_attr"], g1["global_attr"] + g2["global_attr"])
    for k in [
        "senders",
        "receivers",
        "n_nodes",
        "n_edges",
        "n_graphs",
        "global_reps_for_edges",
        "global_reps_for_nodes",
    ]:
        tf.debugging.assert_equal(out[k], g1[k])


def test_concat_tensordicts_concatenates_last_dim():
    t1 = _make_td(with_global=True)
    t2 = _make_td(with_global=True)

    out = _concat_tensordicts(t1, t2)

    assert out["nodes"].shape == (2, 4)
    assert out["edges"].shape == (3, 4)
    assert out["global_attr"].shape == (1, 4)

    tf.debugging.assert_equal(out["nodes"], tf.concat([t1["nodes"], t2["nodes"]], axis=-1))
    tf.debugging.assert_equal(out["edges"], tf.concat([t1["edges"], t2["edges"]], axis=-1))
    tf.debugging.assert_equal(
        out["global_attr"], tf.concat([t1["global_attr"], t2["global_attr"]], axis=-1)
    )


def test_concat_tensordicts_without_global_attr():
    t1 = _make_td(with_global=False)
    t2 = _make_td(with_global=False)

    out = _concat_tensordicts(t1, t2)

    assert "global_attr" not in out
    assert out["nodes"].shape == (2, 4)
    assert out["edges"].shape == (3, 4)


def test_zero_graph_default_state_size():
    g = _make_td(with_global=True)

    out = _zero_graph(g)

    assert out["nodes"].shape == g["nodes"].shape
    assert out["edges"].shape == g["edges"].shape
    assert out["global_attr"].shape == g["global_attr"].shape
    assert np.allclose(out["nodes"].numpy(), 0.0)
    assert np.allclose(out["edges"].numpy(), 0.0)
    assert np.allclose(out["global_attr"].numpy(), 0.0)


def test_zero_graph_explicit_state_size_and_tf_variant_parity():
    g = _make_td(with_global=True)

    out_static = _zero_graph(g, state_size=5)
    out_trace = _zero_graph_tf(g, state_size=5)

    assert out_static["nodes"].shape == (2, 5)
    assert out_static["edges"].shape == (3, 5)
    assert out_static["global_attr"].shape == (1, 5)
    tf.debugging.assert_equal(out_static["nodes"], out_trace["nodes"])
    tf.debugging.assert_equal(out_static["edges"], out_trace["edges"])
    tf.debugging.assert_equal(out_static["global_attr"], out_trace["global_attr"])


def test_zero_graph_tf_default_state_size():
    g = _make_td(with_global=True)

    out = _zero_graph_tf(g)

    assert out["nodes"].shape == g["nodes"].shape
    assert out["edges"].shape == g["edges"].shape
    assert out["global_attr"].shape == g["global_attr"].shape
    np.testing.assert_allclose(out["nodes"].numpy(), 0.0)
    np.testing.assert_allclose(out["edges"].numpy(), 0.0)
    np.testing.assert_allclose(out["global_attr"].numpy(), 0.0)


def test_slice_conc_tensordict_splits_feature_fields():
    td = _make_td(with_global=True)
    td["nodes"] = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32)
    td["edges"] = tf.constant(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=tf.float32
    )
    td["global_attr"] = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)

    first, second = _slice_conc_tensordict(td, [1, 2], [2, 1], [1, 2])

    tf.debugging.assert_equal(first["nodes"], td["nodes"][:, :1])
    tf.debugging.assert_equal(second["nodes"], td["nodes"][:, 1:3])
    tf.debugging.assert_equal(first["edges"], td["edges"][:, :2])
    tf.debugging.assert_equal(second["edges"], td["edges"][:, 2:3])
    tf.debugging.assert_equal(first["global_attr"], td["global_attr"][:, :1])
    tf.debugging.assert_equal(second["global_attr"], td["global_attr"][:, 1:3])
