# Copyright 2021 Charilaos K. Mylonas
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Notes:
# -------
# This code contains the main logic for the full GraphNet computation.
# Some ideas (and sometimes source code) were taken from
# here: https://github.com/deepmind/graph_nets/blob/64771dff0d74ca8e77b1f1dcd5a7d26634356d61/graph_nets/
#


# python imports
import os
from enum import Enum
from collections import OrderedDict

import keras
from keras.layers import Input, Dense, Layer

# Try to load tfp - of not just print something to let the user know that
# they need to do their own probabilsitic ML stuff.
try:
    import tensorflow_probability as tfp

    tfd = tfp.distributions
except ImportError:
    print(
        "Info: Tensorflow probability (tfp) is not available. Some functionality depending on legacy tfp will not be directly available."
    )

import numpy as np

from .tfgnns_datastructures import Graph, GraphTuple, make_graph_tuple_from_graph_list
from . import backend_ops


# The following stores how larger is the aggregation operation
# compared to the message output. For instance mean_max is
# 2 x [dimension of incoming messages]
AGG_TO_MESSAGE_DICT = {
    "mean": 1,
    "max": 1,
    "min": 1,
    "sum": 1,
    "mean_max": 2,
    "mean_max_min": 3,
    "mean_max_min_sum": 4,
}


def _maybe_make_tuple(_v):
    if not isinstance(_v, tuple):
        if isinstance(_v, int):
            return (_v,)
        if isinstance(_v, list):
            return tuple(_v)
    return _v


def unsorted_segment_min_or_zero(
    values, indices, num_groups, name="unsorted_segment_min_or_zero"
):
    """Aggregates information using elementwise min.
    Segments with no elements are given a "min" of zero instead of the most
    positive finite value possible (which is what `tf.math.unsorted_segment_min`
    would do).
    Args:
      values: A `Tensor` of per-element features.
      indices: A 1-D `Tensor` whose length is equal to `values`' first dimension.
      num_groups: A `Tensor`.
      name: (string, optional) A name for the operation.
    Returns:
      A `Tensor` of the same type as `values`.
    """
    del name
    return backend_ops.segment_min_or_zero(values, indices, num_groups)


def unsorted_segment_max_or_zero(
    values, indices, num_groups, name="unsorted_segment_max_or_zero"
):
    """Aggregates information using elementwise max.
    Segments with no elements are given a "max" of zero instead of the most
    negative finite value possible (which is what `tf.math.unsorted_segment_max`
    would do).
    Args:
      values: A `Tensor` of per-element features.
      indices: A 1-D `Tensor` whose length is equal to `values`' first dimension.
      num_groups: A `Tensor`.
      name: (string, optional) A name for the operation.
    Returns:
      A `Tensor` of the same type as `values`.
    """
    del name
    return backend_ops.segment_max_or_zero(values, indices, num_groups)


# Loading some self-assets:
try:
    html_asset_path = os.environ["TFGNNS_HTML_ASSETS"]
    with open(os.path.join(html_asset_path, "pretty_print_accordion.css"), "r") as f:
        ACCORDION_CSS = f.read()
    with open(os.path.join(html_asset_path, "accordion.js")) as f:
        ACCORDION_JS = f.read()
except (OSError, KeyError):
    print(
        "Some assets failed to load!\nSome pretty printing functionality may be unavailable."
    )


def _instantiate_gamma(t, NParams_=1):
    if "tfd" not in locals():
        raise Exception(
            "You tried to call a function that depends on a legacy version of tensorflow probability. Please install a compatible version or refactor the code to use newer tfp versions."
        )
    return tfd.Gamma(
        concentration=t[..., 0:NParams_], rate=t[..., NParams_ : 2 * NParams_]
    )


def _split_tensor_dict_to_repar(
    td, nlatent_nodes_or_all, nlatent_edges=None, nlatent_global=None
):
    """
    splits a tensor dictionary that corresponds to a graphTuple to mean and std parametrization.
    if only `nlatent_nodes_or_all` is defined, then all dimensions are the same.

    """
    nlatent_nodes = nlatent_nodes_or_all
    if nlatent_edges is None:
        nlatent_edges = nlatent_nodes_or_all

    if nlatent_global is None:
        nlatent_global = nlatent_nodes_or_all

    td_mean = td.copy()
    td_std = td.copy()

    td_mean["edges"] = td["edges"][:, :nlatent_edges]
    td_mean["nodes"] = td["nodes"][:, :nlatent_nodes]
    td_mean["global_attr"] = td["global_attr"][:, :nlatent_global]

    td_std["edges"] = td["edges"][:, nlatent_edges:]
    td_std["nodes"] = td["nodes"][:, nlatent_nodes:]
    td_std["global_attr"] = td["global_attr"][:, nlatent_global:]
    return td_mean, td_std


EDGE_FUNCTION_INPUTS = [
    "global_state",
    "sender_node_state",
    "receiver_node_state",
    "edge_state",
]


class EdgeInput(Enum):
    GLOBAL_STATE = "global_state"
    SENDER_NODE_STATE = "sender_node_state"
    RECEIVER_NODE_STATE = "receiver_node_state"
    EDGE_STATE = "edge_state"


NODE_FUNCTION_INPUTS = ["node_state", "edge_state_agg", "global_state"]


class NodeInput(Enum):
    GLOBAL_STATE = "global_state"
    NODE_STATE = "node_state"
    EDGE_AGG_STATE = "edge_state_agg"


GLOBAL_FUNCTION_INPUTS = ["global_state", "edge_state_agg", "node_state_agg"]


class GlobalInput(Enum):
    GLOBAL_STATE = "global_state"
    EDGE_AGG_STATE = "edge_state_agg"
    NODE_AGG_STATE = "node_state_agg"


class GraphNet:
    """One GraphNet computation block.

    A ``GraphNet`` instance wraps edge, node, and optional global update
    functions and evaluates them over either object graphs or tensor-dict graph
    tuples.
    """

    def __init__(
        self,
        edge_function=None,
        node_function=None,
        global_function=None,
        edge_aggregation_function=None,
        node_to_global_aggregation_function=None,
        graph_independent=False,
        use_global_input=False,
        name=None,
    ):
        """Initialize a GraphNet block.

        Args:
            edge_function: Keras model for edge updates.
            node_function: Keras model for node updates.
            global_function: Optional Keras model for global updates.
            edge_aggregation_function: Aggregation used for edge-to-node and
                edge-to-global reductions in non-segment paths.
            node_to_global_aggregation_function: Aggregation used for
                node-to-global reductions.
            graph_independent: If ``True``, disable message passing and apply
                independent edge/node/global transforms.
            use_global_input: Whether global features are provided as model
                inputs.
            name: Optional block name.
        """
        self.edge_aggregation_function, self.edge_aggregation_function_seg = [
            None,
            None,
        ]
        self.is_graph_independent = graph_independent  # should come first.
        self.name = name

        if use_global_input:
            if not graph_independent:
                assert node_to_global_aggregation_function is not None
            if graph_independent:
                assert global_function is not None
            assert global_function is not None

        self.edge_function = edge_function
        self.edge_input_dict, self.node_input_dict, self.global_input_dict = [
            {},
            {},
            {},
        ]
        if self.edge_function is not None:
            self.scan_edge_function()  # checking for consistency among some inputs and keeping track of the inputs to the edge function.

        self.node_function = node_function
        if self.node_function is not None:
            self.scan_node_function()  # checking for consistency among some inputs and keeping track of the inputs to the node function.

        self.global_function = global_function
        self.node_to_global_aggregation_function = node_to_global_aggregation_function
        if self.global_function is not None:
            self.scan_global_function()

        if graph_independent and edge_aggregation_function is not None:
            Exception(
                "Edge-aggregation functions do not make sense in graph-independent blocks! Check your model creation code for errors."
            )

        self.has_seg_aggregator_edge_to_global = False
        self.has_seg_aggregator_edge_to_node = False
        self.has_seg_aggregator_node_to_global = False

        if edge_aggregation_function is not None:
            try:
                len_ea = len(edge_aggregation_function)
            except TypeError:
                len_ea = 1

            if len_ea > 1:
                # Has segment reduction version:
                self.edge_aggregation_function_seg = edge_aggregation_function[1]
                self.has_seg_aggregator_node_to_edge = True
                self.has_seg_aggregator_edge_to_global = True
                self.edge_aggregation_function = edge_aggregation_function[0]
            else:
                self.edge_aggregation_function = edge_aggregation_function

        if node_to_global_aggregation_function is not None:
            try:
                len(node_to_global_aggregation_function)
                self.has_seg_aggregator_node_to_global = True
            except TypeError:
                self.has_seg_aggregator_node_to_global = False

        if self.edge_function is not None:
            self.edge_input_size = self.edge_function.inputs[0].shape[
                1
            ]  # input dimension 1 of edge mlp is the edge state size by convention.

        self.weights = self._weights()

    @staticmethod
    def make_from_path(path):
        graph_functions = GraphNet.load_graph_functions(path)
        return GraphNet(**graph_functions)

    def scan_edge_to_node_aggregation_function(self, fn):
        """
        Scans inputs & outputs of agg. function and keeps track of them for subsequent computation.
        Throws an error if the aggregation is not compatible with the rest of the defined GN functions.
        """
        print("Warning: Empty implementation!")
        return 1

    def scan_edge_function(self):
        """
        Edge function signature (wether it has or has not an input) is inferred by the naming of the inputs.
        Scans the inputs of the edge function to keep track of which graph variables the edge functions uses - throws errors for cases that don't make sense.
        Creates a dict that resolves this correspondence for the evaluation.
        """
        function_input_names = [i.name for i in self.edge_function.inputs]
        # Make sure we have only the expected input state names:

        if not all(
            [
                any([ef in f for ef in EDGE_FUNCTION_INPUTS])
                for f in function_input_names
            ]
        ):
            ef_inputs = "\n" + "  \n".join(function_input_names) + "\n"
            ef_possible_inputs_str = "\n  " + "   \n".join(EDGE_FUNCTION_INPUTS) + "\n"
            Exception(
                "The edge function states should contain any of the following: %s. The entered edge-function inputs were: %s Cannot create GraphNet with this edge function!"
                % (ef_possible_inputs_str, ef_inputs)
            )

        # A dictionary having a correspondence of possible and available edge_inputs:

        for ef in EDGE_FUNCTION_INPUTS:
            for f in function_input_names:
                if ef in f:
                    self.edge_input_dict.update({ef: f})

        if self.is_graph_independent and any(
            [
                k in ["global_state", "sender_node_state", "receiver_node_state"]
                for k in self.edge_input_dict.keys()
            ]
        ):
            ef_inputs_str = "\n  " + "\n  ".join(function_input_names) + "\n"
            Exception(
                "Graph independent networks cannot have node sender , node receiver or global states as inputs! The entered edge function has the following inputs: %s"
                % ef_inputs_str
            )

        # The following help TF to avoid retracing tf.functions blocks.
        (
            self.uses_edge_state,
            self.uses_sender_node_state,
            self.uses_receiver_node_state,
        ) = [False, False, False]

        if EdgeInput.EDGE_STATE.value in self.edge_input_dict.keys():
            self.uses_edge_state = True

        if EdgeInput.SENDER_NODE_STATE.value in self.edge_input_dict.keys():
            self.uses_sender_node_state = True

        if EdgeInput.RECEIVER_NODE_STATE.value in self.edge_input_dict.keys():
            self.uses_receiver_node_state = True

    def scan_global_function(self):
        if self.global_function is not None:
            global_fn_inputs = [i.name for i in self.global_function.inputs]
            possible_global_inputs = [
                "node_state_agg",
                "edge_state_agg",
                "global_state",
            ]
            for nn in possible_global_inputs:
                for g_ in global_fn_inputs:
                    if nn in g_:
                        self.global_input_dict.update({nn: g_})

    def scan_node_function(self):
        """
        Basic sanity checks for the node function.
        """
        node_fn_inputs = [i.name for i in self.node_function.inputs]

        possible_node_inputs = ["node_state", "edge_state_agg", "global_state"]
        for nn in possible_node_inputs:
            for n_ in node_fn_inputs:
                if nn in n_:
                    self.node_input_dict.update({nn: n_})

        if len(self.node_input_dict) is None:
            admis_names_str = "\n" + "\n  ".join(possible_node_inputs)
            node_fn_inputs_str = "\n" + "\n  ".join(node_fn_inputs)
            Exception(
                "Node function inputs don't have ANY of the admissible input names! Admissible input names should contain any of the following:%s\nProvided Node function inputs: %s"
                % (admis_names_str, node_fn_inputs_str)
            )

        if self.is_graph_independent and any(
            [k in ["global_state", "edge_state_agg"] for k in self.node_input_dict]
        ):
            node_fn_inputs_str = "\n" + "\n  ".join(node_fn_inputs)
            Exception(
                "You defined the GraphNet as graph independent but provided message-passing related inputs (global_state or edge_state_agg) to the node function! Provided node-function inputs are:%s"
                % (node_fn_inputs_str)
            )

    def get_graphnet_input_shapes(self):
        result = {}
        if self.edge_function is not None:
            result.update(
                {"edge_function": [i.shape for i in self.edge_function.inputs]}
            )
        result.update({"node_function": [i.shape for i in self.node_function.inputs]})
        return result

    def get_graphnet_output_shapes(self):
        result = {}
        if self.edge_function is not None:
            result.update(
                {"edge_function": [i.shape for i in self.edge_function.outputs]}
            )
        result.update({"node_function": [i.shape for i in self.node_function.outputs]})
        return result

    def summary(self):
        print("Summary of all GraphNet functions")

        def print_summary_if_keras_model(mod_, which_mod):
            print(which_mod)
            if isinstance(mod_, keras.Model):
                mod_.summary()
            else:
                print(mod_)
            print("")

        print_summary_if_keras_model(self.node_function, "Node function")
        print_summary_if_keras_model(self.edge_function, "Edge function")
        print_summary_if_keras_model(self.global_function, "Global function")
        if not self.is_graph_independent:
            print_summary_if_keras_model(
                self.edge_aggregation_function, "Edge Agg. function"
            )
        else:
            print("* No aggregation function - Graph-independent network.")

    def _repr_html_(self):
        """
        Print HTML buttons - for jupyter.
        """
        # TODO: Consider moving pretty-print helpers to a dedicated visualization
        # module so core runtime logic remains smaller and easier to maintain.
        s = ""
        s += "<div>"
        # s += '<html>'
        fn_string = ""
        if self.is_graph_independent:
            fn_string = "Graph Indep. "

        s += ""
        s += "<style>" + ACCORDION_CSS + "</style>"

        def _get_html_repr_keras_fn(self_fn, which_fn):
            ss = ""
            ss += '<button class="accordion tfgnnviz">%s function' % which_fn
            _fn_header_string = (
                "("
                + ", ".join([i.name + ":" + str(i.shape[-1]) for i in self_fn.inputs])
                + ") -> %i" % self_fn.output.shape[-1]
            )
            ss += "" + _fn_header_string + "</button>\n"
            stringlist = []
            self_fn.summary(print_fn=lambda x: stringlist.append(x))
            _fn_string = "<pre>" + "\n".join([li for li in stringlist]) + "</pre>\n"

            ss += '<div class="panel">\n'
            ss += _fn_string
            ss += "</div>\n"
            return ss

        details = ""
        if self.node_function is not None and isinstance(self.node_function, keras.Model):
            details += _get_html_repr_keras_fn(self.node_function, "Node")

        if self.edge_function is not None and isinstance(self.edge_function, keras.Model):
            details += _get_html_repr_keras_fn(self.edge_function, "Edge")

        if self.global_function is not None and isinstance(
            self.global_function, keras.Model
        ):
            details += _get_html_repr_keras_fn(self.global_function, "Global")

        hdr_string = "<h4>%sGNN function (@%s)</h4>" % (fn_string, str(id(self)))
        s += "    <div> %s </div>" % hdr_string
        s += "      %s" % details
        # s += '</html>'

        s += "<script>" + ACCORDION_JS + "</script>"

        return s

    def _weights(self):
        """
        returns the weights of all the functions associated with this GN in a vector.

        These are:
          node_function, edge_function, edge_aggregation_function

        """
        all_weights = []
        if self.edge_function is not None:
            all_weights.extend(self.edge_function.weights)
        if self.node_function is not None:
            all_weights.extend(self.node_function.weights)
        if self.global_function is not None:
            all_weights.extend(self.global_function.weights)

        if not self.is_graph_independent:
            if (
                self.edge_aggregation_function is not None
                and not self.is_graph_independent
            ) and hasattr(self.edge_aggregation_function, "weights"):
                # If the edge aggregation function has weights (it is not a simple aggregation like "mean") accum., append the weights:
                all_weights.extend(self.edge_aggregation_function.weights)

        return all_weights

    def __call__(self, graph):
        if not isinstance(graph, GraphTuple):
            return self.graph_eval(graph)
        else:
            return self.graph_tuple_eval(graph)

    def graph_tuple_eval(self, tf_graph_tuple: GraphTuple):
        # This method parallels what the deepmind library does for faster batched computation.
        # The `tf_graph_tuple` contains edge, nodes, n_edges, n_nodes, senders (indices), receivers (indices) and optionally a global variable (first dimension n_graphs).
        # * the "edges" and "nodes" are already stacked into a single tensor
        # * If .from_node or .to_node tensors are needed for the edge computations they are gathered according to the senders and receivers tensors.
        # * the edge function uses the edge state, (optionally) the sender nodes, (optionally) the receiver nodes.
        # * the edge function outputs are aggregated according to the "receivers" tensor to yield the messages.
        #
        # Global state can also be supplied as an argument (if the edge and/or node functions use it)
        #
        # Arguments:
        #  tf_graph_tuple : a GraphTuple object (i.e. a dictionary containing tensors) containing nodes, edges and their connectivity for multiple graphs.
        #
        # Outputs:
        #  A

        tf_graph_tuple.nodes = backend_ops.convert_to_tensor(tf_graph_tuple.nodes)
        tf_graph_tuple.edges = backend_ops.convert_to_tensor(tf_graph_tuple.edges)
        if tf_graph_tuple.global_attr is not None:
            tf_graph_tuple.global_attr = backend_ops.convert_to_tensor(
                tf_graph_tuple.global_attr
            )

        # if the graph is not graph_indep compute the edge-messages with the aggregator.

        # 1) compute the edge functions
        edge_inputs = {}
        if EdgeInput.EDGE_STATE.value in self.edge_input_dict.keys():
            v = tf_graph_tuple.edges
            edge_inputs.update({EdgeInput.EDGE_STATE.value: v})

        if EdgeInput.SENDER_NODE_STATE.value in self.edge_input_dict.keys():
            v = backend_ops.gather(tf_graph_tuple.nodes, tf_graph_tuple.senders, axis=0)
            edge_inputs.update({EdgeInput.SENDER_NODE_STATE.value: v})

        if EdgeInput.RECEIVER_NODE_STATE.value in self.edge_input_dict.keys():
            v = backend_ops.gather(
                tf_graph_tuple.nodes, tf_graph_tuple.receivers, axis=0
            )
            edge_inputs.update({EdgeInput.RECEIVER_NODE_STATE.value: v})

        if EdgeInput.GLOBAL_STATE.value in self.edge_input_dict.keys():
            edge_reps = tf_graph_tuple._global_reps_for_edges
            # edge_reps = []
            # for k, e in enumerate(tf_graph_tuple.n_edges):
            #    edge_reps.extend([k]*e)
            edge_inputs.update(
                {
                    EdgeInput.GLOBAL_STATE.value: backend_ops.gather(
                        tf_graph_tuple.global_attr, edge_reps, axis=0
                    )
                }
            )

        tf_graph_tuple.edges = self.edge_function(edge_inputs)

        # 2) Aggregate the messages (unsorted segment sums etc):
        node_inputs = {}
        if NodeInput.EDGE_AGG_STATE.value in self.node_input_dict.keys():
            if self.has_seg_aggregator_node_to_edge:
                max_seq = backend_ops.first_dim(tf_graph_tuple.nodes)
                edge_to_nodes_messages = self.edge_aggregation_function_seg(
                    tf_graph_tuple.edges, tf_graph_tuple.receivers, max_seq
                )
            else:
                raise Exception("Not Implemented!")
            node_inputs.update({NodeInput.EDGE_AGG_STATE.value: edge_to_nodes_messages})

        # 3) Compute node function:
        if NodeInput.NODE_STATE.value in self.node_input_dict.keys():
            node_inputs.update({NodeInput.NODE_STATE.value: tf_graph_tuple.nodes})

        if NodeInput.GLOBAL_STATE.value in self.node_input_dict.keys():
            node_reps = tf_graph_tuple._global_reps_for_nodes
            node_inputs.update(
                {
                    NodeInput.GLOBAL_STATE.value: backend_ops.gather(
                        tf_graph_tuple.global_attr, node_reps
                    )
                }
            )

        tf_graph_tuple.nodes = self.node_function(node_inputs)

        # 4) Global computation:
        # Aggregate edges to global:
        global_inputs = {}
        if GlobalInput.EDGE_AGG_STATE.value in self.global_input_dict.keys():
            if (
                self.has_seg_aggregator_node_to_edge
            ):  # <- same aggregator for edges-to-nodes and edges-to-global.
                edges_to_global_messages = self.edge_aggregation_function_seg(
                    tf_graph_tuple.edges,
                    tf_graph_tuple._global_reps_for_edges,
                    tf_graph_tuple.n_graphs,
                )
                global_inputs.update(
                    {GlobalInput.EDGE_AGG_STATE.value: edges_to_global_messages}
                )
            else:
                raise Exception("Not Implemented!")

        if GlobalInput.NODE_AGG_STATE.value in self.global_input_dict.keys():
            if self.has_seg_aggregator_node_to_global:
                nodes_to_global_messages = self.node_to_global_aggregation_function[1](
                    tf_graph_tuple.nodes,
                    tf_graph_tuple._global_reps_for_nodes,
                    tf_graph_tuple.n_graphs,
                )
                global_inputs.update(
                    {GlobalInput.NODE_AGG_STATE.value: nodes_to_global_messages}
                )

        if GlobalInput.GLOBAL_STATE.value in self.global_input_dict.keys():
            global_inputs.update(
                {GlobalInput.GLOBAL_STATE.value: tf_graph_tuple.global_attr}
            )
        if len(global_inputs) > 0:
            global_attr = self.global_function(global_inputs)
            tf_graph_tuple.assign_global(global_attr)

        return tf_graph_tuple
        # no changes in Graph topology - nothing else to do!

    def edge_block(
        self,
        edges=None,
        nodes=None,
        senders=None,
        receivers=None,
        n_edges=None,
        n_nodes=None,
        global_attr=None,
        global_reps_for_edges=None,
        global_reps_for_nodes=None,
        n_graphs=None,
    ):

        edge_inputs = {}
        if EdgeInput.EDGE_STATE.value in self.edge_input_dict.keys():
            v = edges
            edge_inputs.update({EdgeInput.EDGE_STATE.value: v})

        if EdgeInput.SENDER_NODE_STATE.value in self.edge_input_dict.keys():
            v = backend_ops.gather(nodes, senders, axis=0)
            edge_inputs.update({EdgeInput.SENDER_NODE_STATE.value: v})

        if EdgeInput.RECEIVER_NODE_STATE.value in self.edge_input_dict.keys():
            v = backend_ops.gather(nodes, receivers, axis=0)
            edge_inputs.update({EdgeInput.RECEIVER_NODE_STATE.value: v})

        if EdgeInput.GLOBAL_STATE.value in self.edge_input_dict.keys():
            edge_reps = global_reps_for_edges
            edge_inputs.update(
                {
                    EdgeInput.GLOBAL_STATE.value: backend_ops.gather(
                        global_attr, edge_reps, axis=0
                    )
                }
            )

        edges = self.edge_function(edge_inputs)
        return edges

    def node_block(
        self,
        edges=None,
        nodes=None,
        senders=None,
        receivers=None,
        n_edges=None,
        n_nodes=None,
        global_attr=None,
        global_reps_for_edges=None,
        global_reps_for_nodes=None,
        n_graphs=None,
    ):
        # 2) Aggregate the messages (unsorted segment sums etc):
        node_inputs = OrderedDict()
        # 3) Compute node function:
        if NodeInput.NODE_STATE.value in self.node_input_dict.keys():
            node_inputs.update({NodeInput.NODE_STATE.value: nodes})

        if NodeInput.GLOBAL_STATE.value in self.node_input_dict.keys():
            node_inputs.update(
                {
                    NodeInput.GLOBAL_STATE.value: backend_ops.gather(
                        global_attr, global_reps_for_nodes
                    )
                }
            )

        if NodeInput.EDGE_AGG_STATE.value in self.node_input_dict.keys():
            if self.has_seg_aggregator_edge_to_global:
                max_seq = backend_ops.first_dim(nodes)
                edge_to_nodes_messages = self.edge_aggregation_function_seg(
                    edges, receivers, max_seq
                )
            else:
                raise Exception("Not Implemented!")
            node_inputs.update({NodeInput.EDGE_AGG_STATE.value: edge_to_nodes_messages})

        nodes = self.node_function(node_inputs)

        return nodes

    def global_block(
        self,
        edges=None,
        nodes=None,
        senders=None,
        receivers=None,
        n_edges=None,
        n_nodes=None,
        global_attr=None,
        global_reps_for_edges=None,
        global_reps_for_nodes=None,
        n_graphs=None,
    ):

        global_inputs = {}
        if n_graphs is None:
            n_graphs = backend_ops.first_dim(global_attr)

        if GlobalInput.EDGE_AGG_STATE.value in self.global_input_dict.keys():
            if (
                self.has_seg_aggregator_edge_to_global
            ):  # <- same aggregator for edges-to-nodes and edges-to-global.
                edges_to_global_messages = self.edge_aggregation_function_seg(
                    edges, global_reps_for_edges, n_graphs
                )
                global_inputs.update(
                    {GlobalInput.EDGE_AGG_STATE.value: edges_to_global_messages}
                )
            else:
                raise Exception("Not Implemented!")
        if GlobalInput.NODE_AGG_STATE.value in self.global_input_dict.keys():
            if self.has_seg_aggregator_node_to_global:
                nodes_to_global_messages = self.node_to_global_aggregation_function[1](
                    nodes, global_reps_for_nodes, n_graphs
                )
                global_inputs.update(
                    {GlobalInput.NODE_AGG_STATE.value: nodes_to_global_messages}
                )

        if GlobalInput.GLOBAL_STATE.value in self.global_input_dict.keys():
            global_inputs.update({GlobalInput.GLOBAL_STATE.value: global_attr})

        if len(global_inputs) > 0:
            global_attr = self.global_function(global_inputs)
        return global_attr

    def eval_tensor_dict(self, d):
        """
        For better performance this uses a dictionary of all the necessary tensors
        rather than a GraphTuple. A graph tuple is easilly transformed to a dictionary
        of tensors by `_graphtuple_to_tensor_dict` function.

        `d` is an ordered dictionary containing the following:
          'edges','nodes','senders','receivers','n_edges','n_nodes','global_attr','global_reps_for_edges','global_reps_for_nodes'

        """
        d_ = d.copy()
        for k, v in d_.items():
            if v is None:
                continue
            d_[k] = backend_ops.convert_to_tensor(v)
        if self.edge_function is not None:
            new_edges = self.edge_block(**d_)
            d_["edges"] = new_edges
        if self.node_function is not None:
            new_nodes = self.node_block(**d_)
            d_["nodes"] = new_nodes
        if self.global_function is not None:
            new_global = self.global_block(**d_)
            d_["global_attr"] = new_global
        return d_

    def graph_eval(self, graph: Graph, **kwargs):
        """
        Evaluate a single Graph by routing through GraphTuple execution.

        Legacy `safe` / `batched` modes were removed in favor of the unified
        GraphTuple path, which is the supported runtime route.
        """
        if not isinstance(graph, Graph):
            raise TypeError("graph_eval expects a Graph instance")
        if "eval_mode" in kwargs or "return_messages" in kwargs:
            raise ValueError(
                "'safe'/'batched' eval modes were removed; use graph_eval(graph) or graph_tuple_eval(...) instead."
            )

        gt = make_graph_tuple_from_graph_list([graph.copy()])
        gt_out = self.graph_tuple_eval(gt)
        return gt_out.get_graph(0)

    def save(self, path):
        """
        Save the model. Iterates of the keras models required and saves them in a folder.

        Arguments:
          path: the path to save to. Creates it if it does not exist.
        """
        functions = [
            self.node_function,
            self.edge_aggregation_function,
            self.edge_function,
        ]
        path_labels = ["node_function", "edge_aggregation_function", "edge_function"]
        import os

        if not os.path.exists(path):
            os.makedirs(path)

        for model_fcn, label in zip(functions, path_labels):
            if model_fcn is not None:
                d_ = os.path.join(path, label)
                model_fcn.save(d_ + ".keras")

    @staticmethod
    def load_graph_functions(path):
        """
        Returns a list of loaded graph functions.
        """
        function_rel_paths = [
            "node_function",
            "edge_aggregation_function",
            "edge_function",
        ]
        functions = {}

        if not os.path.exists(path):
            print("path does not exist.")
            assert 0

        os.listdir(
            path
        )  # the path should have appropriately named folders that correspond to the diffferent graph functions. All are keras models.
        for function_name in function_rel_paths:
            d_ = os.path.join(path, function_name) + ".keras"
            if not os.path.exists(d_):
                print(
                    "path %s does not exist! Function %s will not be constructed."
                    % (d_, function_name)
                )

            else:
                model_fcn = keras.models.load_model(
                    d_,
                    custom_objects={
                        "SimpleAggLayerDense": SimpleAggLayerDense,
                        "SimpleAggLayerSparse": SimpleAggLayerSparse,
                    },
                )
                functions.update({function_name: model_fcn})
                print("loading %s" % (d_))
        return functions

    def load(self, path):
        """
        Load a model from disk. If the model is already initialized the current graphnet functions are simply overwritten.
        If the model is un-initialized, this is called from a static method (factory method) to make a new object with consistent properties.

        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"path does not exist: {path}")
        loaded = GraphNet.load_graph_functions(path)
        self.node_function = loaded.get("node_function")
        self.edge_aggregation_function = loaded.get("edge_aggregation_function")
        self.edge_function = loaded.get("edge_function")
        self.weights = self._weights()


def make_mlp(units, input_tensor_list, output_shape, activation="relu", **kwargs):
    """
    A default method for making a small MLP.
    Concatenates named inputs provided in a list. The inputs are named even though they are provided in a list.
    This naming is used from the evaluation method (self.graph_tuple_eval). This is less error prone to trying to
    keep a specific ordering for the inputs.
    """
    if len(input_tensor_list) > 1:
        edge_function_input = keras.layers.concatenate(input_tensor_list, axis=1)
    else:
        assert isinstance(input_tensor_list, list)
        edge_function_input = input_tensor_list[
            0
        ]  # essentially a list of a single tensor.

    if "activate_last_layer" in kwargs.keys():
        act_last_layer = kwargs["activate_last_layer"]
    else:
        act_last_layer = False

    if "layernorm_last_layer" in kwargs.keys():
        layernorm_last_layer = kwargs["layernorm_last_layer"]
    else:
        layernorm_last_layer = False

    # A workaround to keep track of created weights easily:
    class DenseMaker:
        def __init__(self):
            self.idx_local_layer = 0

        def make(self, *args, **kwargs):
            kwargs["name"] = "dense_%i" % self.idx_local_layer
            self.idx_local_layer += 1
            return Dense(*args, **kwargs)

    dense_maker = DenseMaker()
    if not isinstance(units, list):
        # If the input is not a list, it is assumed to be an integer.
        # Then a two-layer MLP is constructed.
        y = dense_maker.make(units, use_bias=False)(edge_function_input)
        y = dense_maker.make(units, activation=activation)(y)
        y = dense_maker.make(units, activation=activation)(y)

        if act_last_layer:
            y = dense_maker.make(output_shape[0], activation=activation)(y)
        else:
            y = dense_maker.make(output_shape[0])(y)

    else:
        if len(units) == 0:
            y = dense_maker.make(output_shape[0], use_bias=False)(edge_function_input)
        else:
            y = edge_function_input
            for kk, u in enumerate(units):
                if kk == 0:
                    y = dense_maker.make(u, use_bias=False)(y)
                else:
                    y = dense_maker.make(u)(y)

    if layernorm_last_layer:
        y = keras.layers.LayerNormalization()(y)

    if "model_name" in kwargs.keys():
        # name= kwargs['model_name']
        name = kwargs["model_name"].replace("/", "-")
    else:
        name = None

    return keras.Model(inputs=input_tensor_list, outputs=y, name=name)


def make_node_mlp(
    units,
    edge_message_input_shape=None,
    node_state_input_shape=None,
    global_state_input_shape=None,
    node_emb_size=None,
    use_global_input=False,
    use_edge_state_agg_input=True,
    graph_indep=False,
    use_node_state_input=True,
    activation="relu",
    **kwargs,
):

    if use_edge_state_agg_input and graph_indep:
        raise Exception(
            "Requested a GraphIndep graphnet node function but speciffied use of node input! This is inconsistent."
        )
    if use_global_input and graph_indep:
        raise Exception(
            "Requested a GraphIndep graphnet node function but speciffied use of global input! This is inconsistent."
        )

    all_inputs = []
    if use_node_state_input:
        node_state_in = Input(
            shape=_maybe_make_tuple(node_state_input_shape),
            name=NodeInput.NODE_STATE.value,
        )
        all_inputs.append(node_state_in)

    if use_edge_state_agg_input:
        agg_edge_state_in = Input(
            shape=_maybe_make_tuple(edge_message_input_shape),
            name=NodeInput.EDGE_AGG_STATE.value,
        )
        all_inputs.append(agg_edge_state_in)

    if use_global_input:
        global_state_in = Input(
            shape=_maybe_make_tuple(global_state_input_shape),
            name=NodeInput.GLOBAL_STATE.value,
        )
        all_inputs.append(global_state_in)

    kwargs["model_name"] = "node_fn-model"
    return make_mlp(units, all_inputs, node_emb_size, activation=activation, **kwargs)


def make_global_mlp(
    units,
    global_in_size=None,
    global_emb_size=None,
    node_in_size=None,
    edge_in_size=None,
    use_node_agg_input=True,
    use_edge_agg_input=True,
    use_global_state_input=True,
    node_to_global_agg=None,
    graph_indep=False,
    activation="relu",
    **kwargs,
):
    """
    Always uses the global state input. May use node/edge aggregator (full GraphNet case)
    """

    if graph_indep and (use_node_agg_input or use_edge_agg_input):
        raise Exception(
            "Requested a Graph independent GraphNet global function but speciffied use of node and/or edge aggregated input! This is inconsistent."
        )

    if graph_indep and ((node_in_size is not None) or (edge_in_size is not None)):
        str_msg = ""
        if node_in_size is not None:
            str_msg += "`node_in_size` [%i]" % node_in_size
        if edge_in_size is not None:
            str_msg += "`edge_in_size` [%i]" % edge_in_size
        raise Exception(
            "You have defined shapes for %s and a graph indep. MLP. This is not allowed (check the GN factory method for errors)."
            % str_msg
        )

    if graph_indep:
        if global_in_size is None:
            raise Exception(
                "You requested a graph independent global block but the block does not have a global input shape! If you would like to create a graph-to-global block with no global inputs but global outputs, you need to set `graph_indep` to False."
            )

    global_inputs_list = []
    if use_global_state_input:
        if global_in_size is None:
            raise ValueError(
                "You need to provide an input size to construct the global MLP! You provided `None`."
            )

        global_state_in = Input(
            shape=_maybe_make_tuple(global_in_size), name=GlobalInput.GLOBAL_STATE.value
        )
        global_inputs_list.append(global_state_in)

    if use_node_agg_input:
        node_agg_state = Input(
            shape=_maybe_make_tuple(node_in_size), name=GlobalInput.NODE_AGG_STATE.value
        )
        global_inputs_list.append(node_agg_state)

    if use_edge_agg_input:
        edge_agg_state = Input(
            shape=_maybe_make_tuple(edge_in_size), name=GlobalInput.EDGE_AGG_STATE.value
        )
        global_inputs_list.append(edge_agg_state)

    kwargs["model_name"] = "glob_fn-model"
    return make_mlp(
        units, global_inputs_list, global_emb_size, activation=activation, **kwargs
    )


def make_edge_mlp(
    units,
    edge_state_input_shape=None,
    sender_node_state_output_shape=None,
    global_to_edge_state_size=None,
    receiver_node_state_shape=None,
    edge_output_state_size=None,
    use_edge_state=True,
    use_sender_out_state=True,
    use_receiver_state=True,
    use_global_state=False,
    graph_indep=False,
    activation="relu",
    **kwargs,
):
    """
    When this is a graph-independent edge function, the node states from the sender and receiver are not used.
    As the make_node_mlp, it uses a list of named keras.Input layers.
    """
    tensor_in_list = []

    if use_edge_state:
        edge_state_in = Input(
            shape=_maybe_make_tuple(edge_state_input_shape),
            name=EdgeInput.EDGE_STATE.value,
        )
        tensor_in_list.append(edge_state_in)

    if use_sender_out_state:
        node_state_sender_out = Input(
            shape=_maybe_make_tuple(sender_node_state_output_shape),
            name=EdgeInput.SENDER_NODE_STATE.value,
        )
        tensor_in_list.append(node_state_sender_out)

    if use_receiver_state:
        node_state_receiver_in = Input(
            shape=_maybe_make_tuple(receiver_node_state_shape),
            name=EdgeInput.RECEIVER_NODE_STATE.value,
        )
        tensor_in_list.append(node_state_receiver_in)

    if use_global_state:
        global_state_in = Input(
            shape=_maybe_make_tuple(global_to_edge_state_size),
            name=EdgeInput.GLOBAL_STATE.value,
        )
        tensor_in_list.append(global_state_in)

    kwargs["model_name"] = "edge_fn-model"
    if graph_indep:
        try:
            assert use_sender_out_state is False
        except AssertionError:
            ValueError(
                "The receiver and sender nodes for graph-independent blocks "
                "should not be used as inputs to the edge function! It was "
                "attempted to create an edge function for a graph-indep. "
                "block containing receiver and sender states as inputs."
            )
        ## Building the edge MLP:
        tensor_input_list = [edge_state_in]
        return make_mlp(
            units,
            tensor_input_list,
            edge_output_state_size,
            activation=activation,
            **kwargs,
        )
    else:
        ## Building the edge MLP:
        return make_mlp(
            units,
            tensor_in_list,
            edge_output_state_size,
            activation=activation,
            **kwargs,
        )


DICT_AGG = {
    "mean": (backend_ops.reduce_mean, backend_ops.segment_mean),
    "sum": (backend_ops.reduce_sum, backend_ops.segment_sum),
    "max": (backend_ops.reduce_max, unsorted_segment_max_or_zero),
    "min": (backend_ops.reduce_min, unsorted_segment_min_or_zero),
}


class SimpleAggLayerDense(Layer):
    """Creates a simple aggregator layer (dense)"""

    def __init__(self, agg_type="mean", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agg_type = agg_type

    def call(self, x):
        _agg = DICT_AGG[self.agg_type][0]
        return _agg(x, 0)


class SimpleAggLayerSparse(Layer):
    """Creates a sparse aggregator layer (for use with GraphTuples)"""

    def __init__(self, agg_type="mean", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agg_type = agg_type

    def call(self, x):
        _agg = DICT_AGG[self.agg_type][1]
        return _agg(x, 0)


def make_keras_simple_agg(input_size, agg_type):
    """
    For consistency I'm making this a keras model (easier saving/loading)
    This is for the "naive" graphNet evaluators. There is a fully batched
    aggregator with segment sums that should be preferred.

    parameters:
       input_size : the size of the expected input (mainly useful to enforce consistency)
       agg_type   : ['mean','sum','min','max']
    """
    dict_agg = {
        "mean": (backend_ops.reduce_mean, backend_ops.segment_mean),
        "sum": (backend_ops.reduce_sum, backend_ops.segment_sum),
        "max": (backend_ops.reduce_max, unsorted_segment_max_or_zero),
        "min": (backend_ops.reduce_min, unsorted_segment_min_or_zero),
    }

    # will create both models - the "unsorted segment sum" type (for high performance graph tuple computation) and the safe/slow/easy aggregator.
    x = Input(
        shape=_maybe_make_tuple(input_size), name="edge_messages"
    )  # for "segment" aggregators this needs also a bunch of indices!
    # aggs = dict_agg[agg_type]
    # y = aggs[0](x,0)
    y_basic = SimpleAggLayerDense(agg_type=agg_type)(x)
    m_basic = keras.Model(
        inputs=x, outputs=y_basic, name="basic_%s_aggregator" % agg_type
    )

    m_seg = dict_agg[agg_type][1]
    return m_basic, m_seg


# TODO: Make the aggregators composable
def make_mean_max_agg(input_size):
    """
    A mean and a max aggregator appended together. This was is useful for some special use-cases.

    Inpsired by:
      Corso, Gabriele, et al. "Principal neighbourhood aggregation for graph nets." arXiv preprint arXiv:2004.05718 (2020).
    """
    x = Input(shape=input_size, name="edge_messages")
    v1 = SimpleAggLayerDense(agg_type="mean")(x)
    v2 = SimpleAggLayerDense(agg_type="max")(x)
    agg_out = keras.layers.Concatenate()([v1, v2])
    m_basic = keras.Model(inputs=x, outputs=agg_out, name="basic_meanmax_aggregator")

    def tf_function_agg(x, recv, max_seq):
        v1, v2 = (
            backend_ops.segment_mean(x, recv, max_seq),
            backend_ops.segment_max(x, recv, max_seq),
        )
        agg_ss = backend_ops.concat([v1, v2], axis=-1)
        return agg_ss

    return m_basic, tf_function_agg


def make_mean_max_min_agg(input_size):
    """
    A mean, a max and a min aggregator appended together. This was is useful for some special use-cases.

    Inpsired by:
      Corso, Gabriele, et al. "Principal neighbourhood aggregation for graph nets." arXiv preprint arXiv:2004.05718 (2020).
    """
    x = Input(shape=input_size, name="edge_messages")
    # v1 = tf.reduce_mean(x,0)
    # v2 = tf.reduce_max(x,0)
    # v3 = tf.reduce_min(x,0)
    # agg_out = tf.concat([v1,v2, v3],axis = -1)
    v1 = SimpleAggLayerDense(agg_type="mean")(x)
    v2 = SimpleAggLayerDense(agg_type="max")(x)
    v3 = SimpleAggLayerDense(agg_type="min")(x)
    agg_out = keras.layers.Concatenate()([v1, v2, v3])
    m_basic = keras.Model(
        inputs=x, outputs=agg_out, name="basic_meanmaxmin_aggregator"
    )

    def tf_function_agg(x, recv, max_seq):
        v1, v2, v3 = (
            backend_ops.segment_mean(x, recv, max_seq),
            unsorted_segment_max_or_zero(x, recv, max_seq),
            unsorted_segment_min_or_zero(x, recv, max_seq),
        )
        agg_ss = backend_ops.concat([v1, v2, v3], axis=-1)
        return agg_ss

    return m_basic, tf_function_agg


def make_mean_max_min_sum_agg(input_size):
    x = keras.Input(shape=input_size)
    v1 = SimpleAggLayerDense(agg_type="mean")(x)
    v2 = SimpleAggLayerDense(agg_type="max")(x)
    v3 = SimpleAggLayerDense(agg_type="min")(x)
    v4 = SimpleAggLayerDense(agg_type="sum")(x)
    agg_out = keras.layers.Concatenate()([v1, v2, v3, v4])
    m_basic = keras.Model(
        inputs=x, outputs=agg_out, name="basic_meanmaxminsum_aggregator"
    )

    def tf_function_agg(x, recv, max_seq):
        v1, v2, v3, v4 = (
            backend_ops.segment_mean(x, recv, max_seq),
            backend_ops.segment_max(x, recv, max_seq),
            backend_ops.segment_min(x, recv, max_seq),
            backend_ops.segment_sum(x, recv, max_seq),
        )
        agg_ss = backend_ops.concat([v1, v2, v3, v4], axis=-1)
        return agg_ss

    return m_basic, tf_function_agg


def _aggregation_function_factory(input_shape, agg_type="mean"):
    """
    A factory method to create aggregators for graphNets.
    Somehow I will have to adapt this for unsorted segment reductions (like DeepMind does) because it's much faster.
    Implemented aggregators:
    ----------------------
      sum
      mean
      max
      min
      mean_max
      mean_max_min

    todo:
    ---------------------
      pna_unsc
      std
      moment

    """
    # Each lambda creates a "basic" and a "segment" (sparse) aggregator.
    agg_type_dict = {
        "mean": lambda input_shape: make_keras_simple_agg(input_shape, "mean"),
        "sum": lambda input_shape: make_keras_simple_agg(input_shape, "sum"),
        "max": lambda input_shape: make_keras_simple_agg(input_shape, "max"),
        "min": lambda input_shape: make_keras_simple_agg(input_shape, "min"),
        "mean_max": lambda input_shape: make_mean_max_agg(
            input_shape
        ),  # compound aggregator (appending aggregators)
        "mean_max_min": lambda input_shape: make_mean_max_min_agg(input_shape),
        "mean_max_min_sum": lambda input_shape: make_mean_max_min_sum_agg(input_shape),
    }

    try:
        aggregators = agg_type_dict[agg_type](input_shape)
        return aggregators
    except TypeError:
        aggregators = agg_type_dict[agg_type]((input_shape,))
        return aggregators


def make_mlp_graphnet_functions(
    units,
    node_input_size,
    node_output_size,
    edge_input_size=None,
    edge_output_size=None,
    create_global_function=False,
    global_input_size=None,
    global_output_size=None,
    use_global_input=False,
    use_global_to_edge=False,  # <- if edge mlp takes a global input.
    use_global_to_node=False,  # <- if node mlp takes global input (input GraphTuple should have a globa field.
    node_mlp_use_edge_state_agg_input=True,
    graph_indep=False,
    message_size="auto",
    aggregation_function="mean",
    node_to_global_aggr_fn=None,
    edge_to_global_aggr_fn=None,
    activation="relu",
    activate_last_layer=False,
    **kwargs,
):
    """Build callables required to construct a :class:`GraphNet` block.

    Args:
        units: Width specification used when creating edge/node/global MLPs.
        node_input_size: Input node feature size.
        node_output_size: Output node feature size.
        edge_input_size: Optional input edge feature size. Defaults to
            ``node_input_size``.
        edge_output_size: Optional output edge feature size. Defaults to
            ``node_output_size``.
        create_global_function: If ``True``, create a global update model.
        global_input_size: Optional input global feature size.
        global_output_size: Optional output global feature size.
        use_global_input: If ``True``, consume ``global_attr`` from inputs.
        use_global_to_edge: If ``True``, include input global state in the edge
            model inputs.
        use_global_to_node: If ``True``, include input global state in the node
            model inputs.
        node_mlp_use_edge_state_agg_input: If ``True``, node MLP consumes
            aggregated edge messages.
        graph_indep: If ``True``, create a graph-independent block (no message
            passing).
        message_size: Edge message size, or ``"auto"`` to infer from selected
            aggregation.
        aggregation_function: Aggregation mode used for message passing.
        node_to_global_aggr_fn: Optional override for node-to-global
            aggregation.
        edge_to_global_aggr_fn: Optional override for edge-to-global
            aggregation.
        activation: Activation used for created MLPs.
        activate_last_layer: Whether to apply activation on final MLP layer.
        **kwargs: Forwarded to low-level MLP factory helpers.

    Returns:
        Dictionary of GraphNet constructor keyword arguments, including
        ``edge_function``, ``node_function``, optional ``global_function``, and
        aggregation callables.

    Raises:
        ValueError: If global state is requested by edge/node models while
            ``use_global_input`` is ``False``.
    """

    node_to_global_aggr_fn = (
        aggregation_function
        if node_to_global_aggr_fn is None
        else node_to_global_aggr_fn
    )
    edge_to_global_aggr_fn = (
        aggregation_function
        if edge_to_global_aggr_fn is None
        else edge_to_global_aggr_fn
    )

    node_input = node_input_size
    if edge_input_size is None:
        edge_input_size = node_input_size

    edge_input_size = node_input_size if (edge_input_size is None) else edge_input_size
    edge_output_size = (
        node_output_size if (edge_output_size is None) else edge_output_size
    )

    if (not use_global_input) and (use_global_to_edge or use_global_to_node):
        raise ValueError(
            "You defined `use_global_input` [False] but still request using it as an edge or node input! Please set 'use_global_input' to [True] if you would like to use the global state of the input graph tuple."
        )

    if use_global_input:
        global_to_edge_input_size = global_input_size if use_global_to_edge else None
        global_to_node_input_size = global_input_size if use_global_to_node else None

    global_to_node_input_size = global_input_size if use_global_to_node else None

    def agg_to_message(block_output):
        return {
            "mean": block_output,
            "max": block_output,
            "min": block_output,
            "sum": block_output,
            "mean_max": block_output * 2,
            "mean_max_min": block_output * 3,
            "mean_max_min_sum": block_output * 4,
        }

    ################# Edge function creation
    if message_size == "auto" and not graph_indep:
        edge_output_message_size = edge_output_size
        node_input_message_size = agg_to_message(edge_output_size)[aggregation_function]
        node_to_global_message_size = agg_to_message(node_output_size)[
            aggregation_function
        ]
        edge_to_global_message_size = agg_to_message(edge_output_size)[
            aggregation_function
        ]
    else:
        edge_output_message_size = edge_output_size
        node_input_message_size = edge_output_size
        node_to_global_message_size = node_output_size
        edge_to_global_message_size = edge_output_size

    edge_mlp_args = {
        "edge_state_input_shape": (edge_input_size,),
        "sender_node_state_output_shape": (
            node_input,
        ),  # this has to be compatible with the output of the aggregator.
        "edge_output_state_size": (
            edge_output_message_size,
        ),  # <- in this factory method edge state is the same as message size
        "receiver_node_state_shape": (node_input,),
    }

    edge_mlp_args.update({"graph_indep": graph_indep})
    edge_mlp_args.update({"use_global_state": use_global_to_edge})
    if use_global_input:
        edge_mlp_args.update({"global_to_edge_state_size": global_to_edge_input_size})
    edge_mlp_args.update({"activation": activation})
    edge_mlp_args.update({"activate_last_layer": activate_last_layer})

    edge_mlp = make_edge_mlp(units, **edge_mlp_args)

    ################# Node function creation
    if not graph_indep:
        input_shape = [None, *edge_mlp.outputs[0].shape[1:]]
        agg_fcn = _aggregation_function_factory(
            input_shape, aggregation_function
        )  # First dimension - incoming edge index
    else:
        agg_fcn = None
        node_mlp_use_edge_state_agg_input = False
        if "use_state_agg_input" in kwargs:
            if kwargs["use_edge_state_agg_input"]:
                raise ValueError(
                    "You tried to construct a node function which accepts aggregated edge messages (with use_edge_state_agg_input == True) and at the same time you defined that the network you are constructing is Graph independent! This is inconsistent - check your factory method options!"
                )

    node_mlp_args = {
        "edge_message_input_shape": (node_input_message_size,),
        "node_state_input_shape": (node_input,),
        "node_emb_size": (node_output_size,),
    }
    node_mlp_args.update({"graph_indep": graph_indep})
    node_mlp_args.update({"global_state_input_shape": global_to_node_input_size})
    if "use_edge_state_agg_input" not in kwargs.keys():
        node_mlp_args.update(
            {"use_edge_state_agg_input": node_mlp_use_edge_state_agg_input}
        )

    node_mlp_args.update(
        {"use_global_input": use_global_to_node}
    )  # <- this is from the input global! (refer to the GraphNets paper - that refers to the part that "global" goes to nodes and edges in the block.)
    node_mlp_args.update({"activation": activation})
    node_mlp_args.update({"activate_last_layer": activate_last_layer})

    node_mlp = make_node_mlp(units, **node_mlp_args, **kwargs)

    ################ Global function creation

    ## Aggregation function:

    node_to_global_agg = None
    if not graph_indep:
        n_input_shape = [None, *node_mlp.outputs[0].shape[1:]]
        node_to_global_agg = _aggregation_function_factory(
            n_input_shape, node_to_global_aggr_fn
        )

    global_mlp_params = {}
    global_mlp_params.update({"use_global_input": use_global_input})
    if global_input_size is not None:
        global_mlp_params.update({"global_in_size": (global_input_size,)})  # -#
        try:
            assert use_global_input
        except AssertionError:
            ValueError(
                "When defining that the GN block accepts a 'global' input you "
                "must define the shape of that input! You provided 'None'"
            )

    if global_output_size is not None:
        global_mlp_params.update({"global_emb_size": (global_output_size,)})

    if not graph_indep and global_output_size is not None:
        global_mlp_params.update({"node_in_size": (node_to_global_message_size,)})  # -#
        global_mlp_params.update({"edge_in_size": (edge_to_global_message_size,)})  # -#

    if not graph_indep:
        global_mlp_params.update({"use_node_agg_input": True})
        global_mlp_params.update({"use_edge_agg_input": True})
    else:
        global_mlp_params.update({"use_node_agg_input": False})
        global_mlp_params.update({"use_edge_agg_input": False})

    global_mlp_params.update({"use_global_state_input": use_global_input})
    global_mlp_params.update({"activate_last_layer": activate_last_layer})
    global_mlp_params.update({"graph_indep": graph_indep})

    if create_global_function:
        assert global_output_size is not None
        if use_global_input:  #
            assert (
                global_input_size is not None
            )  # The case where the GT has a global. It is allowed to be None when the global is ignored for the global MLP computation.
        global_mlp = make_global_mlp(units, **global_mlp_params, **kwargs)
    else:
        global_mlp = None

    return {
        "edge_function": edge_mlp,
        "node_function": node_mlp,
        "global_function": global_mlp,
        "edge_aggregation_function": agg_fcn,
        "node_to_global_aggregation_function": node_to_global_agg,  # <- the constructor for GNs does not support different aggs. for edges yet.
        "graph_independent": graph_indep,
        "use_global_input": use_global_input,
    }  # Can be passed directly  to the GraphNet construction with **kwargs


def make_graph_to_graph_and_global_functions(
    units,
    node_or_core_input_size,
    global_output_size,
    node_or_core_output_size=None,
    edge_output_size=None,
    edge_input_size=None,
    aggregation_function="mean",
    **kwargs,
):
    """Create a graph-to-graph-plus-global GraphNet function bundle.

    This wrapper builds a block that does not consume input globals but still
    produces global outputs by aggregating node and edge states.

    Args:
        units: Width specification used for constructed MLPs.
        node_or_core_input_size: Input node feature size.
        global_output_size: Output global feature size.
        node_or_core_output_size: Optional output node feature size.
        edge_output_size: Optional output edge feature size.
        edge_input_size: Optional input edge feature size.
        aggregation_function: Aggregation mode used in message passing.
        **kwargs: Forwarded to :func:`make_mlp_graphnet_functions`.

    Returns:
        Dictionary of constructor arguments for :class:`GraphNet`.
    """
    if node_or_core_output_size is None:
        node_or_core_output_size = global_output_size

    if edge_output_size is None:
        edge_output_size = global_output_size

    if edge_input_size is None:
        edge_input_size = node_or_core_input_size

    if global_output_size is None:
        global_output_size = node_or_core_output_size

    assert np.all(
        [
            k not in ["graph_indep", "create_global_function", "use_global_input"]
            for k in kwargs.keys()
        ]
    )
    return make_mlp_graphnet_functions(
        units,
        node_or_core_input_size,
        node_or_core_output_size,
        use_global_input=False,
        edge_input_size=edge_input_size,
        edge_output_size=edge_output_size,
        create_global_function=True,
        global_output_size=global_output_size,
        use_global_to_edge=False,  # <- refers to the global input (that is un-used)
        use_global_to_node=False,  # <-     >>   >>  >>
        graph_indep=False,
        aggregation_function=aggregation_function,
        **kwargs,
    )


def make_graph_indep_graphnet_functions(
    units,
    node_or_core_input_size,
    node_or_core_output_size=None,
    edge_input_size=None,
    edge_output_size=None,
    global_input_size=None,
    global_output_size=None,
    aggregation_function="mean",
    create_global_function=True,
    use_global_input=True,
    **kwargs,
):
    """Create GraphNet callables for a graph-independent block.

    Args:
        units: Width specification used for created MLPs.
        node_or_core_input_size: Input node feature size.
        node_or_core_output_size: Optional output node feature size.
        edge_input_size: Optional input edge feature size.
        edge_output_size: Optional output edge feature size.
        global_input_size: Optional input global feature size.
        global_output_size: Optional output global feature size.
        aggregation_function: Aggregation mode (retained for interface
            consistency).
        create_global_function: If ``True``, create global output function.
        use_global_input: If ``True``, consume global input state.
        **kwargs: Forwarded to :func:`make_mlp_graphnet_functions`.

    Returns:
        Dictionary of constructor arguments for :class:`GraphNet` configured in
        graph-independent mode.
    """

    if node_or_core_output_size is None:
        node_or_core_output_size = node_or_core_input_size

    if edge_input_size is None:
        edge_input_size = node_or_core_input_size

    if edge_output_size is None:
        edge_output_size = node_or_core_output_size

    if global_input_size is None:
        global_input_size = node_or_core_input_size

    if global_output_size is None:
        global_output_size = node_or_core_output_size

    if node_or_core_input_size is None:
        raise ValueError(
            "You should provide the GN size of the size of several of the involved states!"
        )

    # Just in case it is called from another wrapper that uses kwargs, check if the named inputs are repeated:
    kwargs_forbidden = [
        "graph_indep",
        "create_global_function",
        "use_global_input",
        "use_global_to_edge",
        "use_global_to_node",
    ]
    assert np.all([k not in kwargs_forbidden for k in kwargs.keys()])
    return make_mlp_graphnet_functions(
        units,
        node_or_core_input_size,
        node_or_core_output_size,
        edge_input_size=edge_input_size,
        edge_output_size=edge_output_size,
        global_output_size=global_output_size,
        global_input_size=global_input_size,
        use_global_input=use_global_input,
        use_global_to_edge=False,
        use_global_to_node=False,
        create_global_function=create_global_function,
        graph_indep=True,
        aggregation_function=aggregation_function,
        **kwargs,
    )


def make_mpnn_graphnet_noglobal_functions(
    units,
    node_or_core_input_size,
    node_or_core_output_size=None,
    edge_input_size=None,
    edge_output_size=None,
    aggregation_function="mean",
    **kwargs,
):
    """Create GraphNet callables for MPNN-style blocks without globals.

    Args:
        units: Width specification used for created MLPs.
        node_or_core_input_size: Input node feature size.
        node_or_core_output_size: Optional output node feature size.
        edge_input_size: Optional input edge feature size.
        edge_output_size: Optional output edge feature size.
        aggregation_function: Aggregation mode used for edge-to-node updates.
        **kwargs: Forwarded to :func:`make_mlp_graphnet_functions`.

    Returns:
        Dictionary of constructor arguments for :class:`GraphNet` with no
        global input/output functions.
    """

    if node_or_core_output_size is None:
        node_or_core_output_size = node_or_core_input_size

    if edge_input_size is None:
        edge_input_size = node_or_core_input_size

    if edge_output_size is None:
        edge_output_size = node_or_core_output_size

    if node_or_core_input_size is None:
        raise ValueError(
            "You should provide the GN size of the size of several of the involved states!"
        )

    # Just in case it is called from another wrapper that uses kwargs, check if the named inputs are repeated:
    kwargs_forbidden = [
        "graph_indep",
        "create_global_function",
        "use_global_input",
        "global_input_size",
        "global_output_size",
        "global_input_size",
        "edge_input_size",
        "edge_output_size",
    ]
    assert np.all([k not in kwargs_forbidden for k in kwargs.keys()])
    return make_mlp_graphnet_functions(
        units,
        node_or_core_input_size,
        node_or_core_output_size,
        edge_input_size=edge_input_size,
        edge_output_size=edge_output_size,
        global_output_size=None,
        global_input_size=None,
        use_global_input=False,
        use_global_to_edge=False,
        use_global_to_node=False,
        create_global_function=False,
        aggregation_function=aggregation_function,
        **kwargs,
    )


def make_full_graphnet_functions(
    units,
    node_or_core_input_size,
    node_or_core_output_size=None,
    edge_input_size=None,
    edge_output_size=None,
    global_input_size=None,
    global_output_size=None,
    aggregation_function="mean",
    **kwargs,
):
    """Create GraphNet callables for a full message-passing block.

    Args:
        units: Width specification used for created MLPs.
        node_or_core_input_size: Input node feature size.
        node_or_core_output_size: Optional output node feature size.
        edge_input_size: Optional input edge feature size.
        edge_output_size: Optional output edge feature size.
        global_input_size: Optional input global feature size.
        global_output_size: Optional output global feature size.
        aggregation_function: Aggregation mode for message passing.
        **kwargs: Forwarded to :func:`make_mlp_graphnet_functions`.

    Returns:
        Dictionary of constructor arguments for :class:`GraphNet` configured
        with edge, node, and global update functions.
    """

    if node_or_core_output_size is None:
        node_or_core_output_size = node_or_core_input_size

    if edge_input_size is None:
        edge_input_size = node_or_core_input_size

    if edge_output_size is None:
        edge_output_size = node_or_core_output_size

    if global_input_size is None:
        global_input_size = node_or_core_input_size

    if global_output_size is None:
        global_output_size = node_or_core_output_size

    if node_or_core_input_size is None:
        raise ValueError(
            "You should provide the GN size of the size of several of the involved states!"
        )

    # Just in case it is called from another wrapper that uses kwargs, check if the named inputs are repeated:
    kwargs_forbidden = [
        "graph_indep",
        "create_global_function",
        "use_global_input",
        "global_output_size",
        "global_input_size",
        "edge_input_size",
        "edge_output_size",
    ]
    assert np.all([k not in kwargs_forbidden for k in kwargs.keys()])
    return make_mlp_graphnet_functions(
        units,
        node_or_core_input_size,
        node_or_core_output_size,
        edge_input_size=edge_input_size,
        edge_output_size=edge_output_size,
        global_output_size=global_output_size,
        global_input_size=global_input_size,
        use_global_input=True,
        use_global_to_edge=True,
        use_global_to_node=True,
        create_global_function=True,
        aggregation_function=aggregation_function,
        **kwargs,
    )
