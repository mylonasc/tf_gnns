"""Public package interface for tf_gnns."""

import os

# Set this environment variable first; pretty-print helpers read it on import.
os.environ["TFGNNS_HTML_ASSETS"] = os.path.join(
    __file__.strip("__init__.py"), "assets", "html_css"
)

from .graphnet_utils import (
    _aggregation_function_factory,
    GraphNet,
    make_edge_mlp,
    make_full_graphnet_functions,
    make_global_mlp,
    make_graph_indep_graphnet_functions,
    make_graph_to_graph_and_global_functions,
    make_keras_simple_agg,
    make_mlp_graphnet_functions,
    make_node_mlp,
)
from .lib.gt_ops import (
    _add_gt,
    _assign_add_tensor_dict,
    _concat_tensordicts,
    _copy_structure,
    _slice_conc_tensordict,
    _zero_graph,
    _zero_graph_tf,
)
from .models.graphnet import GraphNetMLP, GraphIndep, GNCellMLP, GraphNetMPNN_MLP
from .tfgnns_datastructures import (
    Edge,
    Graph,
    GraphTuple,
    Node,
    make_graph_tuple_from_graph_list,
)

__all__ = [
    "Graph",
    "GraphTuple",
    "Node",
    "Edge",
    "make_graph_tuple_from_graph_list",
    "GraphNet",
    "make_node_mlp",
    "make_edge_mlp",
    "make_keras_simple_agg",
    "make_mlp_graphnet_functions",
    "make_global_mlp",
    "_aggregation_function_factory",
    "make_full_graphnet_functions",
    "make_graph_indep_graphnet_functions",
    "make_graph_to_graph_and_global_functions",
    "_add_gt",
    "_assign_add_tensor_dict",
    "_concat_tensordicts",
    "_copy_structure",
    "_slice_conc_tensordict",
    "_zero_graph",
    "_zero_graph_tf",
    "GraphNetMLP",
    "GraphIndep",
    "GNCellMLP",
    "GraphNetMPNN_MLP",
    "__version__",
]

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("tf_gnns")
except PackageNotFoundError:
    __version__ = "unknown"
