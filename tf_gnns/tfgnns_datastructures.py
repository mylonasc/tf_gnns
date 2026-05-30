"""Data structures for representing graphs and graph batches.

This module provides object-based graph classes (`Node`, `Edge`, `Graph`) and a
flat batched container (`GraphTuple`) used by high-level GraphNet APIs.
"""

import numpy as np

from . import backend_ops


def _copy_any_ds(val):
    """
    Copy semantics for different datatypes accepted.
    This affects what happens when copying nodes, edges and graphs.
    In order to trace gradients,
    and defines a consistent interface regardless of the input data-structure.
    """
    valout = val
    if isinstance(val, np.ndarray) or isinstance(val, list):
        valout = val.copy()

    if backend_ops.is_tensor(val):
        valout = backend_ops.identity(
            val
        )  # TODO: maybe have a flag to override this? Adding more ops does not always make sense.

    return valout


class Node:
    """Graph node with a feature tensor.

    Args:
        node_attr_tensor: Tensor-like node attributes with at least rank 2.
            The first dimension is usually batch-like in object graph mode.
    """

    def __init__(self, node_attr_tensor):
        if len(node_attr_tensor.shape) < 2:
            raise ValueError(
                "The shape of the input for nodes and edges should have at least 2 dimensions!"
            )
        self.node_attr_tensor = node_attr_tensor
        self.incoming_edges = []
        self.shape = self.node_attr_tensor.shape

    def get_state(self):
        return self.node_attr_tensor

    def set_tensor(self, tensor):
        self.node_attr_tensor = tensor
        self.shape = self.shape = tensor.shape

    def copy(self):
        return Node(_copy_any_ds(self.node_attr_tensor))

    def __add__(self, n):
        return Node(self.node_attr_tensor + n.node_attr_tensor)

    def __sub__(self, n):
        return Node(self.node_attr_tensor - n.node_attr_tensor)


class Edge:
    """Directed edge connecting two :class:`Node` objects.

    Args:
        edge_attr_tensor: Tensor-like edge attributes.
        node_from: Source node.
        node_to: Destination node.
    """

    def __init__(self, edge_attr_tensor, node_from, node_to):
        self.edge_tensor = edge_attr_tensor
        self.node_from = node_from
        self.node_to = node_to
        self.shape = self.edge_tensor.shape

        # Keep a reference to this edge since it is needed for aggregation afterwards.
        node_to.incoming_edges.append(self)

    def set_tensor(self, edge_tensor):
        self.edge_tensor = edge_tensor
        self.shape = edge_tensor.shape

    def copy(self, nodes_correspondence):
        edge_tensor = _copy_any_ds(self.edge_tensor)
        node_from = nodes_correspondence[self.node_from]
        node_to = nodes_correspondence[self.node_to]
        return Edge(edge_tensor, node_from, node_to)

    def __add__(self, edge):
        Exception(
            "Edge addition is not implemented! This is due to potentially unclear semantics. Perform this manually."
        )


class Graph:
    """Object graph made of node and edge instances.

    Args:
        nodes: List of :class:`Node` instances.
        edges: List of :class:`Edge` instances.
        global_attr: Optional graph-level attributes.
        NO_VALIDATION: If ``False``, run connectivity validation checks.
    """

    def __init__(self, nodes, edges, global_attr=None, NO_VALIDATION=True):
        self.nodes = nodes
        self.edges = edges
        self.global_attr = global_attr
        self.has_global = self.global_attr is not None

        if not NO_VALIDATION:
            self.validate_graph()

    def is_equal_by_value(self, g2):
        """
        Checks if the graphs have the same values for node and edge attributes
        """
        is_equal = True
        for n1, n2 in zip(self.nodes, g2.nodes):
            is_equal = is_equal and backend_ops.reduce_all(
                n1.node_attr_tensor == n2.node_attr_tensor
            )

        for e1, e2 in zip(self.edges, g2.edges):
            is_equal = is_equal and backend_ops.reduce_all(
                e1.edge_tensor == e2.edge_tensor
            )
        if self.has_global:
            is_equal = is_equal and (g2.global_attr == self.global_attr)
        return bool(is_equal)

    def compare_connectivity(self, g2):
        """
        Checks if the connectivity of two graphs is the same.
        """
        g1 = self
        nodes_from_match = [
            (g1.nodes.index(e1.node_from) == g2.nodes.index(e2.node_from))
            for e1, e2 in zip(g1.edges, g2.edges)
        ]
        nodes_to_match = [
            (g1.nodes.index(e1.node_to) == g2.nodes.index(e2.node_to))
            for e1, e2 in zip(g1.edges, g2.edges)
        ]
        all_matching = True
        for matches in [*nodes_from_match, *nodes_to_match]:
            all_matching = all_matching and matches
        return all_matching

    @staticmethod
    def validate_graph(self):

        # validate that the edges are all
        for e in self.edges:
            if e.node_from in self.nodes:
                raise AssertionError(
                    "The source node {nn} for edge {ee} is not in the graph!".format(
                        nn=e.node_from, ee=e
                    )
                )
            if e.node_to in self.nodes:
                raise AssertionError(
                    "The destination node {nn} for edge {ee} is not in the graph!".format(
                        nn=e.node_to, ee=e
                    )
                )

    def copy(self):
        # copy attributes of nodes and edges and re-create graph connectivity:
        nodes_coppied = [n.copy() for n in self.nodes]
        nodes_correspondence = {s: c for s, c in zip(self.nodes, nodes_coppied)}
        # Instantiate the new edges:
        coppied_edge_instances = []
        for e in self.edges:
            enew = e.copy(nodes_correspondence)
            coppied_edge_instances.append(enew)
        return Graph(nodes_coppied, coppied_edge_instances)

    def get_subgraph_from_nodes(self, nodes, edge_trimming_mode="+from+to"):
        """Create a subgraph by filtering nodes and incident edges.

        Args:
            nodes: Node subset to keep.
            edge_trimming_mode: Edge filter mode. Supported values are
                ``"+from+to"`` (keep edges where both endpoints are in
                ``nodes``) and ``"-from+to"`` (keep edges where both endpoints
                are not in ``nodes``).

        Returns:
            A new :class:`Graph` with copied nodes and matching copied edges.
        """

        def check_edge_trimming_condition(e_):
            if edge_trimming_mode == "+from+to":
                return (e.node_from in nodes) and (e.node_to in nodes)

            if edge_trimming_mode == "-from+to":
                return (e.node_from not in nodes) and (e.node_to not in nodes)

        sg_nodes_copy = [n.copy() for n in nodes]
        original_copy_nodes_correspondence = {
            n: nc for n, nc in zip(nodes, sg_nodes_copy)
        }
        sg_edges_copy = []
        if len(self.edges) > 0:
            for e in self.edges:
                if check_edge_trimming_condition(e):
                    sg_edges_copy.append(e.copy(original_copy_nodes_correspondence))

        g = Graph(sg_nodes_copy, sg_edges_copy)
        return g

    def __add__(self, graph):
        """
        This should only work with graphs that have compatible node and edge features
        Assumed also that the two graphs have the same connectivity (otherwise this will fail ugly)
        """
        nodes = [nself + n for nself, n in zip(self.nodes, graph.nodes)]
        correspondence = {s: t for s, t in zip(self.nodes, nodes)}
        added_edges = []
        for eself, e in zip(self.edges, graph.edges):
            enew = Edge(
                eself.edge_tensor + e.edge_tensor,
                correspondence[eself.node_from],
                correspondence[eself.node_to],
            )
            added_edges.append(enew)

        return Graph(nodes, added_edges)


def make_graph_tuple_from_graph_list(list_of_graphs):
    """Create a :class:`GraphTuple` from a list of object graphs.

    Args:
        list_of_graphs: List of :class:`Graph` objects with consistent feature
            dimensionality.

    Returns:
        A :class:`GraphTuple` with flattened node/edge tensors and bookkeeping
        vectors (`senders`, `receivers`, `n_nodes`, `n_edges`).

    Notes:
        This helper currently expects node and edge attributes in each input
        graph to have first dimension equal to ``1``.
    """

    # check the first dimension is 1 - instruct to split graphs if not.

    # TODO: Support splitting a list of same graphs with the first dimension of node and edge
    #       features different than one and constructing a GraphTuple. Currently the first
    #       dimension is required to be "1" (but squeezed later on!)
    for g_index, g in enumerate(list_of_graphs):
        problem = ""
        if g.nodes[0].get_state().shape[0] != 1:
            problem += (
                "First size of node attributes should be 1 - found %i "
                % g.edges[0].get_state().shape[0]
            )
        if g.edges[0].edge_tensor.shape[0] != 1:
            problem += (
                "First size of node attributes should be 1 - found %i "
                % g.edges[0].get_state().shape[0]
            )

    # graph_id = [id_ for id_, dummy in enumerate(list_of_graphs)]
    all_edges, all_nodes, n_nodes, n_edges = [[], [], [], []]
    for g in list_of_graphs:
        all_edges.extend(g.edges)
        all_nodes.extend(g.nodes)
        n_nodes.append(len(g.nodes))
        n_edges.append(len(g.edges))

    edge_attr_tensor, nodes_attr_tensor, senders, receivers = [[], [], [], []]
    for e in all_edges:
        edge_attr_tensor.append(e.edge_tensor)
        senders.append(all_nodes.index(e.node_from))
        receivers.append(all_nodes.index(e.node_to))

    for n in all_nodes:
        nodes_attr_tensor.append(n.node_attr_tensor)

    # The 2nd dimension (dimension index 1) should be of size 1 (there is a test in the start of the constructor).
    # The same framework supports efficient computation on graphs of the same topology batched together where the first dimension
    # is the batched size. It is required that such graphs were provided for the construction (or at least the first dimension is "1").
    edges_attr_stacked = backend_ops.squeeze(backend_ops.stack(edge_attr_tensor, 0), 1)
    nodes_attr_stacked = backend_ops.squeeze(
        backend_ops.stack(nodes_attr_tensor, 0), 1
    )
    return GraphTuple(
        nodes_attr_stacked, edges_attr_stacked, senders, receivers, n_nodes, n_edges
    )  # , graph_id)


class GraphTuple:
    """Batched graph representation used by GraphNet tensor-dict paths.

    A ``GraphTuple`` stores all node and edge features in contiguous tensors and
    keeps graph boundaries via `n_nodes` and `n_edges` vectors.

    The ``GraphTuple`` makes multiple smaller graphs appear as a single large
    graph, with contiguous indexing for nodes and edges. This allows fast
    batched computation and takes advantage of default performance
    optimizations in deep learning frameworks.
    """

    def __init__(
        self,
        nodes,
        edges,
        senders,
        receivers,
        n_nodes,
        n_edges,
        global_attr=None,
        global_reps_for_nodes=None,
        global_reps_for_edges=None,
        n_graphs=None,
    ):
        """Initialize a graph batch.

        Args:
            nodes: Tensor-like node feature array with shape ``[sum(n_nodes), d_n]``.
            edges: Tensor-like edge feature array with shape ``[sum(n_edges), d_e]``.
            senders: Sender node indices for each edge. Indices are unique
                across graphs in the flattened representation.
            receivers: Receiver node indices for each edge. Indices are unique
                across graphs in the flattened representation.
            n_nodes: Per-graph node counts.
            n_edges: Per-graph edge counts.
            global_attr: Optional graph-level features of shape
                ``[n_graphs, d_g]``.
            global_reps_for_nodes: Optional precomputed mapping from node rows
                to graph ids.
            global_reps_for_edges: Optional precomputed mapping from edge rows
                to graph ids.
            n_graphs: Optional number of graphs.
        """
        # Sort edges according to receivers and sort receivers:
        assert len(n_nodes) == len(n_edges)

        self.nodes = nodes  # floats tensor
        self.edges = edges  # floats tensor
        self.senders = senders  # integers
        self.receivers = receivers  # integers
        self.n_nodes = n_nodes  # integers
        self.n_edges = n_edges  # integers
        if n_graphs is None:
            self.n_graphs = len(
                self.n_nodes
            )  # assuming the n_nodes is a list containing the number of nodes for each graph.

        self.global_attr = global_attr
        self.has_global = self.global_attr is not None

        graph_indices_nodes = []
        for k_, k in enumerate(self.n_nodes):
            graph_indices_nodes.extend(np.ones(k).astype("int") * k_)

        graph_indices_edges = []
        for k_, k in enumerate(self.n_edges):
            graph_indices_edges.extend(np.ones(k).astype("int") * k_)

        if self.has_global:  # <- default global is "None". If it was provided, set the global variable (together with some aggregator indices for convenience and performance).
            self.assign_global(global_attr)

        self.graph_indices_nodes, self.graph_indices_edges = (
            graph_indices_nodes,
            graph_indices_edges,
        )

        if (global_reps_for_edges is None) and (global_reps_for_nodes is None):
            self.update_reps_for_globals()
        self.n_graphs = len(self.n_nodes)

    def update_reps_for_globals(self):
        """Build helper vectors mapping nodes/edges to graph indices."""
        global_reps_for_edges = []  # <- used to cast the global tensor to a compatible size for the edges.
        for k, e in enumerate(self.n_edges):
            global_reps_for_edges.extend([k] * int(e))
        self._global_reps_for_edges = global_reps_for_edges

        global_reps_for_nodes = []  # <- similarly for nodes:
        for k, e in enumerate(self.n_nodes):
            global_reps_for_nodes.extend([k] * int(e))

        self._global_reps_for_nodes = global_reps_for_nodes

    def assign_global(self, global_attr, check_shape=False):
        """Assign graph-level features.

        Args:
            global_attr: Tensor-like global features.
            check_shape: If ``True``, assert first dimension equals ``n_graphs``.
        """
        self.has_global = True
        if check_shape:
            assert int(backend_ops.first_dim(global_attr)) == self.n_graphs
        self.global_attr = global_attr

    def is_equal_by_value(self, other_graph_tuple):
        v1 = (
            self.edges,
            self.nodes,
            self.receivers,
            self.senders,
            self.n_nodes,
            self.n_edges,
            self.n_graphs,
        )
        v2 = (
            other_graph_tuple.edges,
            other_graph_tuple.nodes,
            other_graph_tuple.receivers,
            other_graph_tuple.senders,
            other_graph_tuple.n_nodes,
            other_graph_tuple.n_edges,
            other_graph_tuple.n_graphs,
        )

        def _equals_or_all_equals(v1_, v2_):
            if isinstance(v1_, list) and isinstance(v2_, list):
                return v1_ == v2_
            if backend_ops.is_tensor(v1_) and backend_ops.is_tensor(v2_):
                return all(v1_ == v2_)
            if isinstance(v1_, np.array) and isinstance(v2_.np.array):
                return all(v1_ == v2_)

        if self.has_global:
            global_same = _equals_or_all_equals(
                other_graph_tuple.global_attr, self.global_attr
            )
            assert other_graph_tuple.has_global
        else:
            global_same = True

        return (
            all([_equals_or_all_equals(v1__, v2__) for v1__, v2__ in zip(v1, v2)])
            and global_same
        )

    def copy(self):
        n = _copy_any_ds(self.nodes)
        e = _copy_any_ds(self.edges)
        s = _copy_any_ds(self.senders)
        r = _copy_any_ds(self.receivers)
        nnodes = _copy_any_ds(self.n_nodes)
        nedges = _copy_any_ds(self.n_edges)
        _copy_any_ds(self.n_graphs)
        return GraphTuple(n, e, s, r, nnodes, nedges, global_attr=self.global_attr)

    def __add__(self, g2):
        nodes = self.nodes + g2.nodes
        edges = self.edges + g2.edges
        s = self.senders
        r = self.receivers
        n_nodes = self.n_nodes
        n_edges = g2.n_edges
        if self.has_global and g2.has_global:
            new_global = self.global_attr + g2.global_attr
            gt = GraphTuple(
                nodes, edges, s, r, n_nodes, n_edges, global_attr=new_global
            )
            gt._global_reps_for_edges = self._global_reps_for_edges
            gt._global_reps_for_nodes = self._global_reps_for_nodes
        else:
            gt = GraphTuple(nodes, edges, s, r, n_nodes, n_edges)

        return gt

    def get_graph(self, graph_index):
        """Extract a single :class:`Graph` from this batch.

        Args:
            graph_index: Zero-based index of the graph to extract.

        Returns:
            A new :class:`Graph` object containing copied node/edge features.
        """
        assert graph_index >= 0
        if graph_index > self.n_graphs:
            raise ValueError(
                "The provided index is larger than the available graphs in this GraphTuple object."
            )

        def get_start_stop_index(sizes_list, index):
            return np.cumsum([0, *sizes_list[0 : index + 1]])[-2:]

        start_idx_nodes, end_idx_nodes = get_start_stop_index(self.n_nodes, graph_index)
        start_idx_edges, end_idx_edges = get_start_stop_index(self.n_edges, graph_index)
        nodes_attrs = self.nodes[start_idx_nodes:end_idx_nodes]
        senders, receivers, edge_attr = [
            v[start_idx_edges:end_idx_edges]
            for v in [self.senders, self.receivers, self.edges]
        ]
        senders = senders - start_idx_nodes
        receivers = receivers - start_idx_nodes
        nodes = [Node(backend_ops.expand_dims(node_attr, axis=0)) for node_attr in nodes_attrs]
        edges = [
            Edge(
                backend_ops.expand_dims(edge_attr_tensor, axis=0),
                nodes[node_from_idx],
                nodes[node_to_idx],
            )
            for edge_attr_tensor, node_from_idx, node_to_idx in zip(
                edge_attr, senders, receivers
            )
        ]
        if self.has_global:
            global_attr = self.global_attr[graph_index]
        else:
            global_attr = None
        return Graph(nodes, edges, global_attr=global_attr)

    def to_tensor_dict(self):
        """Convert this graph batch to a GraphNet tensor dictionary."""
        return _graphtuple_to_tensor_dict(self)


def _graphtuple_to_tensor_dict(gt_):
    """Convert a :class:`GraphTuple` into the tensor-dict graph format.

    Args:
        gt_: Input graph tuple.

    Returns:
        Dictionary with keys expected by high-level GraphNet layers.
    """

    def _tf_constant_or_none(v):
        if v is None:
            return None
        else:
            return backend_ops.convert_to_tensor(v)

    return {
        "edges": _tf_constant_or_none(gt_.edges),
        "nodes": _tf_constant_or_none(gt_.nodes),
        "senders": _tf_constant_or_none(gt_.senders),
        "receivers": _tf_constant_or_none(gt_.receivers),
        "n_edges": _tf_constant_or_none(gt_.n_edges),
        "n_nodes": _tf_constant_or_none(gt_.n_nodes),
        "n_graphs": _tf_constant_or_none(gt_.n_graphs),
        "global_attr": _tf_constant_or_none(gt_.global_attr),
        "global_reps_for_edges": _tf_constant_or_none(gt_._global_reps_for_edges),
        "global_reps_for_nodes": _tf_constant_or_none(gt_._global_reps_for_nodes),
    }
