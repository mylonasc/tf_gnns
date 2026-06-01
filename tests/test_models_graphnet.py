import pytest
import tensorflow as tf

from tf_gnns.models.graphnet import GraphIndep, GraphNetMLP, GraphNetMPNN_MLP
from tf_gnns.tfgnns_datastructures import GraphTuple


def _make_td(with_global=True):
    edges = tf.constant([[0.5, -1.0], [1.5, 2.0], [-0.3, 0.7]], dtype=tf.float32)
    nodes = tf.constant([[1.0, 0.0], [2.0, -1.0], [3.0, 4.0]], dtype=tf.float32)
    senders = tf.constant([0, 1, 2], dtype=tf.int32)
    receivers = tf.constant([1, 2, 0], dtype=tf.int32)
    global_attr = tf.constant([[0.2, -0.4]], dtype=tf.float32) if with_global else None
    gt = GraphTuple(
        nodes,
        edges,
        senders,
        receivers,
        n_nodes=[3],
        n_edges=[3],
        global_attr=global_attr,
    )
    td = gt.to_tensor_dict()
    return td


def test_graphnet_mlp_non_residual_and_recurrent_builds_and_runs():
    td = _make_td(with_global=True)
    model = GraphNetMLP(units=8, core_steps=2, recurrent=True, residual=False)
    out = model(td)

    assert out["nodes"].shape[0] == td["nodes"].shape[0]
    assert out["edges"].shape[0] == td["edges"].shape[0]
    assert out["global_attr"] is not None


def test_graphnet_mpnn_mlp_non_residual_and_recurrent_runs():
    td = _make_td(with_global=False)
    model = GraphNetMPNN_MLP(units=8, core_steps=2, recurrent=True, residual=False)
    out = model(td)

    assert out["nodes"].shape == td["nodes"].shape
    assert out["edges"].shape == td["edges"].shape


def test_graphnet_mlp_respects_explicit_output_feature_sizes():
    td = _make_td(with_global=True)
    model = GraphNetMLP(
        units=8,
        core_steps=2,
        recurrent=False,
        residual=True,
        node_output_size=7,
        edge_output_size=5,
        global_output_size=3,
    )
    out = model(td)

    assert out["nodes"].shape == (td["nodes"].shape[0], 7)
    assert out["edges"].shape == (td["edges"].shape[0], 5)
    assert out["global_attr"].shape == (td["global_attr"].shape[0], 3)


def test_graphnet_mlp_preserves_graph_structure_tensors():
    td = _make_td(with_global=True)
    model = GraphNetMLP(units=8, core_steps=1, recurrent=False, residual=True)
    out = model(td)

    for key in [
        "senders",
        "receivers",
        "n_nodes",
        "n_edges",
        "global_reps_for_edges",
        "global_reps_for_nodes",
        "n_graphs",
    ]:
        tf.debugging.assert_equal(out[key], td[key])


def test_graphnet_mpnn_respects_explicit_output_feature_sizes():
    td = _make_td(with_global=False)
    model = GraphNetMPNN_MLP(
        units=8,
        core_steps=2,
        recurrent=False,
        residual=True,
        node_output_size=6,
        edge_output_size=4,
    )
    out = model(td)

    assert out["nodes"].shape == (td["nodes"].shape[0], 6)
    assert out["edges"].shape == (td["edges"].shape[0], 4)


def test_graphnet_mpnn_ignores_non_graph_keys_in_tensor_dict():
    td = _make_td(with_global=False)
    td["labels"] = tf.constant([1.0], dtype=tf.float32)
    td["metadata"] = tf.constant([123], dtype=tf.int32)
    model = GraphNetMPNN_MLP(units=8, core_steps=1, recurrent=False, residual=True)
    out = model(td)

    assert out["nodes"].shape[0] == td["nodes"].shape[0]
    assert out["edges"].shape[0] == td["edges"].shape[0]


def test_graphindep_builds_and_runs_with_and_without_global():
    td_with_global = _make_td(with_global=True)
    td_no_global = _make_td(with_global=False)

    m_with_global = GraphIndep(units_out=6)
    out_global = m_with_global(td_with_global)
    assert out_global["nodes"].shape[-1] == 6
    assert out_global["edges"].shape[-1] == 6

    m_no_global = GraphIndep(units_out=5)
    with pytest.raises(TypeError):
        m_no_global(td_no_global)
