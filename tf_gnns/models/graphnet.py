from tf_gnns.graphnet_utils import make_full_graphnet_functions, make_graph_indep_graphnet_functions, make_graph_to_graph_and_global_functions, _aggregation_function_factory
from tf_gnns.graphnet_utils import GraphNet 

from tf_gnns.lib.gt_ops import _assign_add_tensor_dict

import tensorflow as tf

class GraphNetMLP(tf.keras.Model):
    """
    A GraphNet with all functions implemented as MLPs
    """
    def __init__(self,
                 units = 32,
                 core_units = None,
                 core_size = None,
                 gi_units = None,
                 core_steps = 1,
                 edge_input_size = None,
                 node_input_size = None,
                 global_input_size = None,
                 edge_output_size = None,
                 node_output_size = None,
                 global_output_size = None,
                 recurrent = False,
                 residual = True,
                 aggregation_function = 'mean',
                 *args, **kwargs):
        super(GraphNetMLP, self).__init__(*args, **kwargs)

        self.aggregation_function = aggregation_function
        if core_units is None:
            core_units = units
        if gi_units is None:
            gi_units = units
        if core_size is None:
            core_size = units

        self.core_units = core_units
        self.gi_units   = gi_units

        self.edge_output_size   = edge_output_size
        self.node_output_size   = node_output_size
        self.global_output_size = global_output_size
        self._is_recurrent      = recurrent
        self.core = core_size
        self.is_built = False
        self.is_residual = residual
        self.core_steps = core_steps

        if edge_input_size is not None  and node_input_size is not None:
            self.build(edge_input_size, node_input_size)
            self.is_built = True

        self.all_weights = []

    def build(self,  d_shapes):

        node_input_size, edge_input_size = d_shapes['nodes'][-1], d_shapes['edges'][-1]

        if 'global_attr' in d_shapes.keys():
            global_input_size = d_shapes['global_attr'][-1]
        else:
            global_input_size = None

        if self.edge_output_size is None:
            self.edge_output_size = edge_input_size
        if self.node_output_size is None:
            self.node_output_size = node_input_size
        if self.global_output_size is None:
            self.global_output_size = global_input_size


        if global_input_size is None:
            self.g_enc_determ = GraphNet(
                **make_graph_to_graph_and_global_functions(self.gi_units,
                                                           node_or_core_input_size=node_input_size,
                                                           node_or_core_output_size=self.core,
                                                           global_output_size=self.core,
                                                           edge_input_size=edge_input_size,
                                                           edge_output_size=self.core,
                                                           aggregation_function= self.aggregation_function))
        else:
            self.g_enc_determ = GraphNet(
                **make_graph_indep_graphnet_functions(self.gi_units,
                                                      node_or_core_input_size = node_input_size,
                                                      node_or_core_output_size= self.core,
                                                      edge_input_size         = edge_input_size,
                                                      global_input_size       = global_input_size,
                                                      global_output_size      = self.core,
                                                      edge_output_size        = self.core))
        g_core_determ_list = [];
        for c_ in range(self.core_steps):
            if (c_ == 0) or ((not self._is_recurrent) and c_ > 0):
                g = GraphNet(**make_full_graphnet_functions(self.core_units,
                                   node_or_core_input_size=self.core))

            g_core_determ_list.append(g) # if it is recurrent, it's multiple times the same network.

        self.g_dec_determ = GraphNet(
            **make_graph_indep_graphnet_functions(self.core_units,
                                            node_or_core_input_size = self.core,
                                            node_or_core_output_size=  self.node_output_size,
                                            edge_input_size         = self.core,
                                            global_input_size       = self.core,
                                            global_output_size      = self.global_output_size,
                                            edge_output_size        = self.edge_output_size,
                                            aggregation_function = self.aggregation_function))

        self.is_built = True

        if self._is_recurrent:
            core_weights = g_core_determ_list[0].weights
        else:
            core_weights = [w.weights for w in g_core_determ_list]

        self.all_weights.extend([*self.g_enc_determ.weights, *core_weights, *self.g_dec_determ.weights])
        self.g_core_determ = g_core_determ_list


    def _repr_html_(self):

        s = ''
        s += "<h> GN Deterministic path containing the following GNs (%s,%s): </h> "%(self.name,id(self))
        if not self.is_built:
            s += "Model not yet built."
            return s

        s += self.g_enc_determ._repr_html_()

        for g_ in self.g_core_determ:
            s += g_._repr_html_()

        s += self.g_dec_determ._repr_html_()
        return s

    @tf.function
    def call(self, g_):
        g_ = self.g_enc_determ.eval_tensor_dict(g_)
        for _gn in self.g_core_determ:
            go_ = _gn.eval_tensor_dict(g_)
            if self.is_residual:
                g_ = _assign_add_tensor_dict(g_, go_)
            else:
                g_ = go_
        g_ = self.g_dec_determ.eval_tensor_dict(g_)
        return g_


class GraphIndep(tf.keras.Model):
    """
    A single graph-independent block.
    """
    def __init__(self, units_out, 
                 gn_mlp_units = [], 
                 node_output_size = None,
                 edge_output_size = None,
                 global_output_size = None,
                *args,**kwargs):
        super(GraphIndep, self).__init__(*args, **kwargs)
        self.units = units_out
        self._gn_mlp_units = gn_mlp_units
        self.is_built = False
        if node_output_size is None:
            node_output_size = units_out
        if edge_output_size is None:
            edge_output_size = units_out
        if global_output_size is None:
            global_output_size = units_out
            
        self.node_output_size  = node_output_size
        self.edge_output_size  = edge_output_size
        self.global_output_size = global_output_size
        
    def build(self,input_shape):
        gnfns = make_graph_indep_graphnet_functions(self._gn_mlp_units,
                                            node_or_core_input_size=input_shape['nodes'][1],
                                            edge_input_size        =input_shape['edges'][1],
                                            global_input_size      =input_shape['global_attr'][1],
                                            node_or_core_output_size = self.node_output_size,
                                            edge_output_size       = self.edge_output_size,
                                            global_output_size     = self.global_output_size)
        
        
        self.gn_graph_indep = GraphNet(**gnfns)
        self.all_weights = self.gn_graph_indep.weights
        self.is_built = True
        
    @tf.function
    def call(self, g_):
        return self.gn_graph_indep.eval_tensor_dict(g_)

