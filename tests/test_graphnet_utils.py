import numpy as np
import pytest
import tensorflow as tf
from pathlib import Path

from tf_gnns import GraphNet
from tf_gnns.graphnet_utils import (
    _aggregation_function_factory,
    make_global_mlp,
    make_graph_indep_graphnet_functions,
    make_mpnn_graphnet_noglobal_functions,
    make_mlp_graphnet_functions,
    make_node_mlp,
    unsorted_segment_max_or_zero,
    unsorted_segment_min_or_zero,
)
from tf_gnns.tfgnns_datastructures import Edge, Graph, GraphTuple, Node


def _make_tensor_dict(with_global=True):
    nodes = tf.constant(
        [[1.0, 0.5], [2.0, -1.0], [0.0, 3.0], [4.0, 2.0]], dtype=tf.float32
    )
    edges = tf.constant(
        [[0.2, 0.7], [1.0, -1.0], [3.0, 1.0], [-0.5, 2.5]], dtype=tf.float32
    )
    senders = tf.constant([0, 1, 2, 3], dtype=tf.int32)
    receivers = tf.constant([1, 1, 3, 2], dtype=tf.int32)
    n_nodes = tf.constant([2, 2], dtype=tf.int32)
    n_edges = tf.constant([2, 2], dtype=tf.int32)
    n_graphs = tf.constant(2, dtype=tf.int32)
    global_reps_for_nodes = tf.constant([0, 0, 1, 1], dtype=tf.int32)
    global_reps_for_edges = tf.constant([0, 0, 1, 1], dtype=tf.int32)
    td = {
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_graphs": n_graphs,
        "global_reps_for_nodes": global_reps_for_nodes,
        "global_reps_for_edges": global_reps_for_edges,
    }
    if with_global:
        td["global_attr"] = tf.constant([[1.0, -1.0], [0.5, 2.0]], dtype=tf.float32)
    else:
        td["global_attr"] = None
    return td


def _to_gt(td):
    return GraphTuple(
        nodes=td["nodes"],
        edges=td["edges"],
        senders=td["senders"],
        receivers=td["receivers"],
        n_nodes=td["n_nodes"],
        n_edges=td["n_edges"],
        global_attr=td["global_attr"],
    )


def test_eval_tensor_dict_matches_graph_tuple_eval_with_global():
    td = _make_tensor_dict(with_global=True)
    gn_args = make_mlp_graphnet_functions(
        16,
        node_input_size=2,
        node_output_size=3,
        edge_input_size=2,
        edge_output_size=4,
        use_global_input=True,
        use_global_to_edge=True,
        use_global_to_node=True,
        create_global_function=True,
        global_input_size=2,
        global_output_size=5,
        graph_indep=False,
        aggregation_function="mean",
    )
    gn = GraphNet(**gn_args)

    out_td = gn.eval_tensor_dict(td.copy())
    out_gt = gn.graph_tuple_eval(_to_gt(td).copy())

    tf.debugging.assert_near(out_td["edges"], out_gt.edges, atol=1e-6, rtol=1e-6)
    tf.debugging.assert_near(out_td["nodes"], out_gt.nodes, atol=1e-6, rtol=1e-6)
    tf.debugging.assert_near(out_td["global_attr"], out_gt.global_attr, atol=1e-6, rtol=1e-6)


def test_eval_tensor_dict_matches_graph_tuple_eval_without_global():
    td = _make_tensor_dict(with_global=False)
    gn_args = make_mlp_graphnet_functions(
        12,
        node_input_size=2,
        node_output_size=2,
        edge_input_size=2,
        edge_output_size=2,
        use_global_input=False,
        use_global_to_edge=False,
        use_global_to_node=False,
        create_global_function=False,
        graph_indep=False,
        aggregation_function="mean",
    )
    gn = GraphNet(**gn_args)

    out_td = gn.eval_tensor_dict(td.copy())
    out_gt = gn.graph_tuple_eval(_to_gt(td).copy())
    tf.debugging.assert_near(out_td["edges"], out_gt.edges, atol=1e-6, rtol=1e-6)
    tf.debugging.assert_near(out_td["nodes"], out_gt.nodes, atol=1e-6, rtol=1e-6)


def test_unsorted_segment_min_max_or_zero_empty_groups_are_zero():
    values = tf.constant([[3.0], [7.0]], dtype=tf.float32)
    indices = tf.constant([0, 2], dtype=tf.int32)
    num_groups = 4

    out_min = unsorted_segment_min_or_zero(values, indices, num_groups)
    out_max = unsorted_segment_max_or_zero(values, indices, num_groups)

    expected = tf.constant([[3.0], [0.0], [7.0], [0.0]], dtype=tf.float32)
    tf.debugging.assert_equal(out_min, expected)
    tf.debugging.assert_equal(out_max, expected)


@pytest.mark.parametrize(
    "agg_type,mult",
    [
        ("mean", 1),
        ("sum", 1),
        ("max", 1),
        ("min", 1),
        ("mean_max", 2),
        ("mean_max_min", 3),
        ("mean_max_min_sum", 4),
    ],
)
def test_aggregation_factory_output_shapes(agg_type, mult):
    dense_agg, seg_agg = _aggregation_function_factory((None, 3), agg_type=agg_type)
    x = tf.constant([[1.0, 2.0, 3.0], [4.0, 1.0, 2.0]], dtype=tf.float32)
    recv = tf.constant([0, 1], dtype=tf.int32)

    dense_out = dense_agg(tf.expand_dims(x, axis=0))
    seg_out = seg_agg(x, recv, 2)

    assert dense_out.shape[-1] == 3 * mult
    assert seg_out.shape[-1] == 3 * mult


def test_make_mlp_graphnet_functions_rejects_global_node_edge_mismatch():
    with pytest.raises(ValueError, match="use_global_input"):
        make_mlp_graphnet_functions(
            8,
            node_input_size=4,
            node_output_size=4,
            edge_input_size=4,
            edge_output_size=4,
            use_global_input=False,
            use_global_to_edge=True,
            use_global_to_node=False,
            create_global_function=False,
            graph_indep=False,
        )


def test_make_mlp_graphnet_functions_rejects_graph_indep_with_edge_messages():
    with pytest.raises(Exception, match="GraphIndep"):
        make_mlp_graphnet_functions(
            8,
            node_input_size=4,
            node_output_size=4,
            graph_indep=True,
            use_edge_state_agg_input=True,
        )


def test_make_node_mlp_rejects_inconsistent_graph_indep_inputs():
    with pytest.raises(Exception, match="GraphIndep"):
        make_node_mlp(
            8,
            edge_message_input_shape=(4,),
            node_state_input_shape=(4,),
            node_emb_size=(4,),
            graph_indep=True,
            use_edge_state_agg_input=True,
        )

    with pytest.raises(Exception, match="GraphIndep"):
        make_node_mlp(
            8,
            edge_message_input_shape=(4,),
            node_state_input_shape=(4,),
            global_state_input_shape=(2,),
            node_emb_size=(4,),
            graph_indep=True,
            use_global_input=True,
            use_edge_state_agg_input=False,
        )


def test_make_global_mlp_rejects_invalid_graph_indep_configurations():
    with pytest.raises(Exception, match="Graph independent"):
        make_global_mlp(
            8,
            global_in_size=(4,),
            global_emb_size=(4,),
            node_in_size=(4,),
            use_node_agg_input=True,
            graph_indep=True,
        )

    with pytest.raises(Exception, match="global input shape"):
        make_global_mlp(
            8,
            global_in_size=None,
            global_emb_size=(4,),
            use_node_agg_input=False,
            use_edge_agg_input=False,
            use_global_state_input=False,
            graph_indep=True,
        )


def test_wrapper_factories_reject_invalid_inputs():
    with pytest.raises(ValueError, match="provide the GN size"):
        make_graph_indep_graphnet_functions(8, node_or_core_input_size=None)

    with pytest.raises(ValueError, match="provide the GN size"):
        make_mpnn_graphnet_noglobal_functions(8, node_or_core_input_size=None)


def test_graph_eval_rejects_removed_legacy_modes():
    node_in = tf.keras.layers.Input(shape=(2,), name="node_state")
    edge_in = tf.keras.layers.Input(shape=(2,), name="edge_state")
    node_fn = tf.keras.Model(inputs=[node_in], outputs=node_in)
    edge_fn = tf.keras.Model(inputs=[edge_in], outputs=edge_in)
    gn = GraphNet(node_function=node_fn, edge_function=edge_fn, graph_independent=True)

    n1 = Node(tf.constant([[1.0, 2.0]], dtype=tf.float32))
    n2 = Node(tf.constant([[3.0, 4.0]], dtype=tf.float32))
    e12 = Edge(tf.constant([[0.5, -0.5]], dtype=tf.float32), n1, n2)
    g = Graph([n1, n2], [e12])

    with pytest.raises(ValueError, match="removed"):
        gn.graph_eval(g, eval_mode="safe")


def test_graphnet_save_load_and_load_method(tmp_path):
    path = Path(tmp_path) / "gn_saved"
    gn_args = make_mlp_graphnet_functions(
        8,
        node_input_size=2,
        node_output_size=2,
        edge_input_size=2,
        edge_output_size=2,
        graph_indep=False,
        use_global_input=False,
        create_global_function=False,
    )
    gn = GraphNet(**gn_args)
    gn.save(str(path))

    loaded_fns = GraphNet.load_graph_functions(str(path))
    assert "node_function" in loaded_fns
    assert "edge_function" in loaded_fns

    gn_loaded = GraphNet.make_from_path(str(path))
    assert len(gn_loaded.weights) == len(gn.weights)

    gn2 = GraphNet(node_function=None, edge_function=None)
    gn2.load(str(path))
    assert gn2.node_function is not None
    assert gn2.edge_function is not None

    with pytest.raises(FileNotFoundError):
        gn2.load(str(path / "missing"))
