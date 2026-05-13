"""Operations on tensor dictionaries that represent graph tuples."""

import tensorflow as tf

GRAPH_TUPLE_STRUCTURE = [
    "senders",
    "receivers",
    "n_nodes",
    "n_edges",
    "global_reps_for_edges",
    "global_reps_for_nodes",
    "n_graphs",
]


def _zero_graph(g_, state_size=None):
    """Create a zero-valued tensor dictionary with the same graph structure.

    Args:
        g_: Input tensor dictionary with graph-tuple bookkeeping fields and
            tensor fields (`nodes`, `edges`, and `global_attr`).
        state_size: Optional feature size used for all output tensors. If
            omitted, output tensor shapes match `g_`.

    Returns:
        A copied tensor dictionary with zero-valued feature tensors.
    """
    g_copy = g_.copy()
    if state_size is None:
        g_copy["nodes"] = tf.zeros_like(g_["nodes"])
        g_copy["edges"] = tf.zeros_like(g_["edges"])
        g_copy["global_attr"] = tf.zeros_like(g_["global_attr"])
    else:
        g_copy["nodes"] = tf.zeros([g_["nodes"].shape[0], state_size], tf.float32)
        g_copy["edges"] = tf.zeros([g_["edges"].shape[0], state_size], tf.float32)
        g_copy["global_attr"] = tf.zeros(
            [g_["global_attr"].shape[0], state_size], dtype=tf.float32
        )

    return g_copy


def _zero_graph_tf(g_, state_size=None):
    """TensorFlow-traceable variant of `_zero_graph`.

    Args:
        g_: Input tensor dictionary.
        state_size: Optional output feature size.

    Returns:
        A copied tensor dictionary with zero-valued feature tensors.
    """
    g_copy = g_.copy()
    if state_size is None:
        g_copy["nodes"] = tf.zeros_like(g_["nodes"])
        g_copy["edges"] = tf.zeros_like(g_["edges"])
        g_copy["global_attr"] = tf.zeros_like(g_["global_attr"])
    else:
        g_copy["nodes"] = tf.zeros([tf.shape(g_["nodes"])[0], state_size], tf.float32)
        g_copy["edges"] = tf.zeros([tf.shape(g_["edges"])[0], state_size], tf.float32)
        g_copy["global_attr"] = tf.zeros(
            [tf.shape(g_["global_attr"])[0], state_size], dtype=tf.float32
        )

    return g_copy


def _add_gt(g1, g2):
    """Add feature tensors from two tensor dictionaries.

    Args:
        g1: First tensor dictionary.
        g2: Second tensor dictionary with matching tensor shapes.

    Returns:
        A new tensor dictionary with `g1` structure and summed features.
    """
    s = _copy_structure(g1)
    nodes_ = g1["nodes"] + g2["nodes"]
    edges_ = g1["edges"] + g2["edges"]
    global_attr_ = g1["global_attr"] + g2["global_attr"]
    s.update({"nodes": nodes_, "edges": edges_, "global_attr": global_attr_})
    return s


def _concat_tensordicts(t1, t2):
    """Concatenate graph feature tensors along the last axis.

    Args:
        t1: First tensor dictionary.
        t2: Second tensor dictionary with matching leading dimensions.

    Returns:
        A tensor dictionary containing concatenated features.
    """
    td_new = _copy_structure(t1)
    td_new["nodes"] = tf.concat([t1["nodes"], t2["nodes"]], axis=-1)
    td_new["edges"] = tf.concat([t1["edges"], t2["edges"]], axis=-1)
    if "global_attr" in t1:
        td_new["global_attr"] = tf.concat(
            [t1["global_attr"], t2["global_attr"]], axis=-1
        )
    return td_new


def _copy_structure(g_):
    """Copy only graph bookkeeping fields from a tensor dictionary.

    Args:
        g_: Input tensor dictionary.

    Returns:
        A dictionary containing only graph structure keys.
    """
    td_new = {k: g_[k] for k in GRAPH_TUPLE_STRUCTURE}
    return td_new


def _assign_add_tensor_dict(d_, od):
    """Add feature tensors from `od` into `d_` in place.

    Args:
        d_: Base tensor dictionary to be updated.
        od: Tensor dictionary providing additive feature tensors.

    Returns:
        The updated `d_` dictionary.
    """
    d_["nodes"] = d_["nodes"] + od["nodes"]
    d_["edges"] = d_["edges"] + od["edges"]
    d_["global_attr"] = d_["global_attr"] + od["global_attr"]
    return d_


def _slice_conc_tensordict(td_, node_slices, edge_slices, glob_slices):
    """Split concatenated graph features into multiple tensor dictionaries.

    Args:
        td_: Input tensor dictionary with concatenated features.
        node_slices: List of node feature widths for each split.
        edge_slices: List of edge feature widths for each split.
        glob_slices: List of global feature widths for each split.

    Returns:
        A list of tensor dictionaries preserving graph structure and slicing
        node/edge/global feature tensors by the provided widths.
    """
    tds_ = []
    assert len(node_slices) == len(edge_slices)
    assert len(edge_slices) == len(glob_slices)
    cni, cei, cgi = [0, 0, 0]
    num_slices = len(node_slices)
    for k in range(num_slices):
        ni, ei, gi = [node_slices[k], edge_slices[k], glob_slices[k]]
        new_tds = _copy_structure(td_)
        new_tds["nodes"] = td_["nodes"][:, cni : cni + ni]
        new_tds["edges"] = td_["edges"][:, cei : cei + ei]
        new_tds["global_attr"] = td_["global_attr"][:, cgi : cgi + gi]
        tds_.append(new_tds)
        cei += ei
        cni += ni
        cgi += gi

    return tds_
