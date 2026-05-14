import numpy as np
import pytest
import keras

from tf_gnns import GraphNet, make_mlp_graphnet_functions
from tf_gnns import backend_ops
from tf_gnns.tfgnns_datastructures import GraphTuple


pytestmark = pytest.mark.skipif(
    keras.backend.backend() != "torch",
    reason="Torch-backend runtime checks only",
)


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    return keras.ops.convert_to_numpy(x)


def _make_td(with_global=True):
    nodes = keras.ops.convert_to_tensor(
        [[1.0, 0.5], [2.0, -1.0], [0.0, 3.0], [4.0, 2.0]], dtype="float32"
    )
    edges = keras.ops.convert_to_tensor(
        [[0.2, 0.7], [1.0, -1.0], [3.0, 1.0], [-0.5, 2.5]], dtype="float32"
    )
    senders = keras.ops.convert_to_tensor([0, 1, 2, 3], dtype="int32")
    receivers = keras.ops.convert_to_tensor([1, 1, 3, 2], dtype="int32")
    n_nodes = [2, 2]
    n_edges = [2, 2]
    global_attr = None
    if with_global:
        global_attr = keras.ops.convert_to_tensor([[1.0, -1.0], [0.5, 2.0]], dtype="float32")

    gt = GraphTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_nodes=n_nodes,
        n_edges=n_edges,
        global_attr=global_attr,
    )
    return gt.to_tensor_dict(), gt


def test_torch_backend_graphtuple_to_tensor_dict_is_backend_tensors():
    td, _ = _make_td(with_global=True)
    for key, value in td.items():
        if value is None:
            continue
        assert backend_ops.is_tensor(value), f"{key} is not a backend tensor"


def test_torch_backend_eval_tensor_dict_matches_graph_tuple_eval():
    td, gt = _make_td(with_global=True)
    gn_args = make_mlp_graphnet_functions(
        12,
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
    out_gt = gn.graph_tuple_eval(gt.copy())

    np.testing.assert_allclose(_to_numpy(out_td["edges"]), _to_numpy(out_gt.edges), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(_to_numpy(out_td["nodes"]), _to_numpy(out_gt.nodes), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(
        _to_numpy(out_td["global_attr"]), _to_numpy(out_gt.global_attr), atol=1e-6, rtol=1e-6
    )
