"""Public package interface for tf_gnns."""

import importlib
import os

# Set this environment variable first; pretty-print helpers read it on import.
os.environ["TFGNNS_HTML_ASSETS"] = os.path.join(
    __file__.strip("__init__.py"), "assets", "html_css"
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
    "SparseGCNConv",
    "SparseGCN",
    "__version__",
]

_LAZY_EXPORTS = {
    "Graph": "tf_gnns.tfgnns_datastructures",
    "GraphTuple": "tf_gnns.tfgnns_datastructures",
    "Node": "tf_gnns.tfgnns_datastructures",
    "Edge": "tf_gnns.tfgnns_datastructures",
    "make_graph_tuple_from_graph_list": "tf_gnns.tfgnns_datastructures",
    "GraphNet": "tf_gnns.graphnet_utils",
    "make_node_mlp": "tf_gnns.graphnet_utils",
    "make_edge_mlp": "tf_gnns.graphnet_utils",
    "make_keras_simple_agg": "tf_gnns.graphnet_utils",
    "make_mlp_graphnet_functions": "tf_gnns.graphnet_utils",
    "make_global_mlp": "tf_gnns.graphnet_utils",
    "_aggregation_function_factory": "tf_gnns.graphnet_utils",
    "make_full_graphnet_functions": "tf_gnns.graphnet_utils",
    "make_graph_indep_graphnet_functions": "tf_gnns.graphnet_utils",
    "make_graph_to_graph_and_global_functions": "tf_gnns.graphnet_utils",
    "_add_gt": "tf_gnns.lib.gt_ops",
    "_assign_add_tensor_dict": "tf_gnns.lib.gt_ops",
    "_concat_tensordicts": "tf_gnns.lib.gt_ops",
    "_copy_structure": "tf_gnns.lib.gt_ops",
    "_slice_conc_tensordict": "tf_gnns.lib.gt_ops",
    "_zero_graph": "tf_gnns.lib.gt_ops",
    "_zero_graph_tf": "tf_gnns.lib.gt_ops",
    "GraphNetMLP": "tf_gnns.models.graphnet",
    "GraphIndep": "tf_gnns.models.graphnet",
    "GNCellMLP": "tf_gnns.models.graphnet",
    "GraphNetMPNN_MLP": "tf_gnns.models.graphnet",
    "SparseGCNConv": "tf_gnns.models.gcn",
    "SparseGCN": "tf_gnns.models.gcn",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'tf_gnns' has no attribute {name!r}")


def __dir__():
    return sorted([*globals(), *_LAZY_EXPORTS])

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("tf_gnns")
except PackageNotFoundError:
    __version__ = "unknown"
