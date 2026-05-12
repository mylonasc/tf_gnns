import numpy as np
import tensorflow as tf

from tf_gnns import Edge, Graph, Node, make_graph_tuple_from_graph_list


def _make_graph(node_offset=0.0, edge_offset=0.0):
    n1 = Node(tf.constant([[1.0 + node_offset, 2.0 + node_offset]], dtype=tf.float32))
    n2 = Node(tf.constant([[3.0 + node_offset, 4.0 + node_offset]], dtype=tf.float32))
    n3 = Node(tf.constant([[5.0 + node_offset, 6.0 + node_offset]], dtype=tf.float32))

    e12 = Edge(tf.constant([[10.0 + edge_offset, 20.0 + edge_offset]], dtype=tf.float32), n1, n2)
    e23 = Edge(tf.constant([[30.0 + edge_offset, 40.0 + edge_offset]], dtype=tf.float32), n2, n3)

    return Graph([n1, n2, n3], [e12, e23])


def test_graph_tuple_roundtrip_preserves_values_and_connectivity():
    graphs_in = [_make_graph(0.0, 0.0), _make_graph(100.0, 200.0)]

    gt = make_graph_tuple_from_graph_list([g.copy() for g in graphs_in])
    graphs_out = [gt.get_graph(i) for i in range(gt.n_graphs)]

    assert len(graphs_out) == len(graphs_in)
    for g_expected, g_out in zip(graphs_in, graphs_out):
        assert len(g_expected.nodes) == len(g_out.nodes)
        assert len(g_expected.edges) == len(g_out.edges)
        for n_expected, n_out in zip(g_expected.nodes, g_out.nodes):
            tf.debugging.assert_equal(n_expected.get_state(), n_out.get_state())
        for e_expected, e_out in zip(g_expected.edges, g_out.edges):
            tf.debugging.assert_equal(e_expected.edge_tensor, e_out.edge_tensor)

        expected_senders = [g_expected.nodes.index(e.node_from) for e in g_expected.edges]
        expected_receivers = [g_expected.nodes.index(e.node_to) for e in g_expected.edges]
        out_senders = [g_out.nodes.index(e.node_from) for e in g_out.edges]
        out_receivers = [g_out.nodes.index(e.node_to) for e in g_out.edges]
        assert out_senders == expected_senders
        assert out_receivers == expected_receivers


def test_graph_tuple_global_reps_and_tensor_dict_invariants():
    graphs = [_make_graph(), _make_graph(10.0, 10.0)]
    gt = make_graph_tuple_from_graph_list(graphs)

    global_attr = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    gt.assign_global(global_attr)
    gt.update_reps_for_globals()

    assert len(gt._global_reps_for_nodes) == int(np.sum(gt.n_nodes))
    assert len(gt._global_reps_for_edges) == int(np.sum(gt.n_edges))
    assert gt._global_reps_for_nodes.count(0) == int(gt.n_nodes[0])
    assert gt._global_reps_for_nodes.count(1) == int(gt.n_nodes[1])
    assert gt._global_reps_for_edges.count(0) == int(gt.n_edges[0])
    assert gt._global_reps_for_edges.count(1) == int(gt.n_edges[1])

    td = gt.to_tensor_dict()
    required_keys = {
        "edges",
        "nodes",
        "senders",
        "receivers",
        "n_edges",
        "n_nodes",
        "n_graphs",
        "global_attr",
        "global_reps_for_edges",
        "global_reps_for_nodes",
    }
    assert required_keys.issubset(td.keys())
    assert int(tf.shape(td["nodes"])[0]) == int(np.sum(gt.n_nodes))
    assert int(tf.shape(td["edges"])[0]) == int(np.sum(gt.n_edges))
    assert int(tf.shape(td["global_attr"])[0]) == gt.n_graphs
