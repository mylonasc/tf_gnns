import numpy as np
import keras

from tf_gnns.models.graphnet import GraphNetMLP, GraphNetMPNN_MLP
from tf_gnns.tfgnns_datastructures import GraphTuple


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    return keras.ops.convert_to_numpy(x)


def test_notebook_graphnet_high_level_flow():
    nodes = keras.ops.convert_to_tensor(
        [[1.0, 0.0], [2.0, -1.0], [3.0, 4.0]], dtype="float32"
    )
    edges = keras.ops.convert_to_tensor(
        [[0.5, -1.0], [1.5, 2.0], [-0.3, 0.7]], dtype="float32"
    )
    senders = keras.ops.convert_to_tensor([0, 1, 2], dtype="int32")
    receivers = keras.ops.convert_to_tensor([1, 2, 0], dtype="int32")
    global_attr = keras.ops.convert_to_tensor([[0.2, -0.4]], dtype="float32")

    _ = GraphTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_nodes=[3],
        n_edges=[3],
        global_attr=global_attr,
    )

    td = {
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "n_nodes": keras.ops.convert_to_tensor([3], dtype="int32"),
        "n_edges": keras.ops.convert_to_tensor([3], dtype="int32"),
        "n_graphs": keras.ops.convert_to_tensor(1, dtype="int32"),
        "global_attr": global_attr,
        "global_reps_for_edges": keras.ops.convert_to_tensor([0, 0, 0], dtype="int32"),
        "global_reps_for_nodes": keras.ops.convert_to_tensor([0, 0, 0], dtype="int32"),
    }

    model_global = GraphNetMLP(
        units=16,
        core_steps=2,
        recurrent=False,
        residual=True,
        node_output_size=6,
        edge_output_size=5,
        global_output_size=4,
    )
    out_global = model_global(td)

    assert tuple(_to_numpy(keras.ops.shape(out_global["nodes"]))) == (3, 6)
    assert tuple(_to_numpy(keras.ops.shape(out_global["edges"]))) == (3, 5)
    assert tuple(_to_numpy(keras.ops.shape(out_global["global_attr"]))) == (1, 4)

    td_no_global = dict(td)
    td_no_global["global_attr"] = None
    model_mpnn = GraphNetMPNN_MLP(
        units=16,
        core_steps=2,
        recurrent=False,
        residual=True,
        node_output_size=6,
        edge_output_size=5,
    )
    out_mpnn = model_mpnn(td_no_global)

    assert tuple(_to_numpy(keras.ops.shape(out_mpnn["nodes"]))) == (3, 6)
    assert tuple(_to_numpy(keras.ops.shape(out_mpnn["edges"]))) == (3, 5)

    for key in [
        "senders",
        "receivers",
        "n_nodes",
        "n_edges",
        "global_reps_for_edges",
        "global_reps_for_nodes",
        "n_graphs",
    ]:
        np.testing.assert_equal(_to_numpy(out_global[key]), _to_numpy(td[key]))
