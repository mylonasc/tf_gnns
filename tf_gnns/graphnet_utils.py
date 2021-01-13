# python imports
import os
from enum import Enum
import inspect
import code

# tensorflow imports:
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential,Model
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalAveragePooling1D, Dropout
import tensorflow_probability as tfp
tfd = tfp.distributions

import numpy as np

#from minigraphnets import Gr3aph, GraphTuple imported at init?
from .datastructures import Graph

def _instantiate_gamma(t, NParams_ = 1):
    return tfd.Gamma(concentration = t[...,0:NParams_], rate = t[...,NParams_:2*NParams_])


EDGE_FUNCTION_INPUTS = ['global_state','sender_node_state','receiver_node_state','edge_state']
class EdgeInput(Enum):
    GLOBAL_STATE = 'global_state'
    SENDER_NODE_STATE = 'sender_node_state'
    RECEIVER_NODE_STATE = 'receiver_node_state'
    EDGE_STATE = 'edge_state'


NODE_FUNCTION_INPUTS = ['node_state','edge_state_agg','global_state']
class NodeInput(Enum):
    NODE_STATE = 'node_state'
    EDGE_AGG_STATE = 'edge_state_agg'
    GLOBAL_STATE = 'global_state'

class GraphNet:
    """
    Input is a graph and output is a graph.
    Encapsulates a GraphNet computation iteration.
    
    Supports model loading and saving (for a single GraphNet)

    Should treat the situations where edge functions do not exist more uniformly.
    Also there is no Special treatment for "globals".
    """
    def __init__(self, edge_function = None, node_function = None, edge_aggregation_function = None, node_to_prob= None, graph_independent = False ):
        """
        A GraphNet class. 
        The constructor expects that edge_function, node_function are Keras models with specially named inputs. The input names 
        of these function are scanned during object construction so they are correctly used when the `GraphNet` is evaluated on some `Graph` or `GraphTuple`.
        
        parameters:
            edge_function             : the edge function (depending on whether the graphnet block is graph 
                                        independent, and whether the source and destinations are used,
                                        this has different input sizes)

            node_function             : the node function (if this is graph independent it has only node inputs)

            edge_aggregation_function : the edge aggregation function used in the non-fully batched evaluation 
                                        modes. ("batched" and "safe"). If it contains two aggregation functions, 
                                        the second one is the "unsorted_segment" variant (for faster computation with GraphTuples)

           node_to_prob               : the function that takes the final graph and returns a tensorflow probability distribution.
                                        can be implemented as a keras layer with a DistributionLambda output. Breaks saving/loading in some cases - untested atm.

        """
        self.is_graph_independent = graph_independent # should come first.

        self.edge_function             = edge_function
        self.scan_edge_function() # checking for consistency among some inputs and keeping track of the inputs to the edge function.

        self.node_function             = node_function
        self.scan_node_function() #checking for consistency among some inputs and keeping track of the inputs to the node function.


        if graph_independent and edge_aggregation_function is not None:
            Exception("On all non-graph independent graphnets an aggregation function should be defined!")

        self.has_seg_aggregator = False
        if edge_aggregation_function is not None:
            if len(edge_aggregation_function ) > 1:
                # Has segment sum version:
                self.edge_aggregation_function_seg = edge_aggregation_function[1]
                self.has_seg_aggregator = True
                self.edge_aggregation_function = edge_aggregation_function[0]
            else:
                self.edge_aggregation_function = edge_aggregation_function        

            self.scan_edge_to_node_aggregation_function(node_function)

        self.node_to_prob_function = node_to_prob # That's sort of special/application speciffic. Maybe remove in the future.
        # Needed to treat the case of no edges.
        # If there are no edges, the aggregated edge state is zero.
        
        if self.edge_function is not None: # a messy hack to regret about later
            self.edge_input_size = self.edge_function.inputs[0].shape[1] # input dimension 1 of edge mlp is the edge state size by convention.

    @staticmethod
    def make_from_path(path):
        graph_functions = GraphNet.load_graph_functions(path)
        return GraphNet(**graph_functions)

    def scan_edge_to_node_aggregation_function(self, fn):
        """
        Scans inputs & outputs of agg. function and keeps track of them for subsequent computation.
        Throws an error if the aggregation is not compatible with the rest of the defined GN functions.
        """
        return 1

    def scan_edge_function(self):
        """
        Edge function signature (wether it has or has not an input) is inferred by the naming of the inputs.
        Scans the inputs of the edge function to keep track of which graph variables the edge functions uses - throws errors for cases that don't make sense.
        Creates a dict that resolves this correspondence for the evaluation.
        """
        function_input_names = [i.name for i in self.edge_function.inputs]
        # Make sure we have only the expected input state names:

        if not all([any([ef in f for ef in EDGE_FUNCTION_INPUTS]) for f in function_input_names]):
            ef_inputs = "\n" + "  \n".join(function_input_names) + "\n"
            ef_possible_inputs_str = "\n  "+"   \n".join(EDGE_FUNCTION_INPUTS) + "\n"
            Exception("The edge function states should contain any of the following: %s. The entered edge-function inputs were: %s Cannot create GraphNet with this edge function!"%(ef_possible_inputs_str,ef_inputs))

        # A dictionary having a correspondence of possible and available edge_inputs:
        self.edge_input_dict= {}
        for ef in EDGE_FUNCTION_INPUTS:
            for f in function_input_names:
                if ef in f:
                    self.edge_input_dict.update({ef:f})

        if self.is_graph_independent and any([k in ['global_state','sender_node_state','receiver_node_state'] for k in self.edge_input_dict.keys()]):
            ef_inputs_str = "\n  " + "\n  ".join(function_input_names) + "\n"
            Exception("Graph independent networks cannot have node sender , node receiver or global states as inputs! The entered edge function has the following inputs: %s"%ef_inputs_str)



        #These are booleans so TF can easilly make tf.functions and not re-trace them.
        self.uses_edge_state, self.uses_sender_node_state, self.uses_receiver_node_state = [False, False, False]

        if EdgeInput.EDGE_STATE.value in self.edge_input_dict.keys():
            self.uses_edge_state = True

        if EdgeInput.SENDER_NODE_STATE.value in self.edge_input_dict.keys():
            self.uses_sender_node_state = True

        if EdgeInput.RECEIVER_NODE_STATE.value in self.edge_input_dict.keys():
            self.uses_receiver_node_state = True

    def scan_node_function(self):
        node_fn_inputs = [i.name for i in self.node_function.inputs]

        possible_node_inputs = ['node_state','edge_state_agg','global_state']
        self.node_input_dict = {}
        for nn in possible_node_inputs:
            for n_ in node_fn_inputs:
                if nn in n_:
                    self.node_input_dict.update({nn: n_})

        if len(self.node_input_dict) is None:
            admis_names_str = "\n" + "\n  ".join(possible_node_inputs)
            node_fn_inputs_str = "\n" + "\n  ".join(node_fn_inputs)
            Exception("Node function inputs don't have ANY of the admissible input names! Admissible input names should contain any of the following:%s\nProvided Node function inputs: %s"%(admis_names_str,node_fn_inputs_str))

        if self.is_graph_independent and any([k in ['global_state','edge_state_agg'] for k in self.node_input_dict]):
            node_fn_inputs_str = "\n" + "\n  ".join(node_fn_inputs)
            Exception("You defined the graphNet as graph independent but provided message-passing related inputs (global_state or edge_state_agg) to the node function! Provided node-function inputs are:%s"%(node_fn_inputs_str))



    def get_graphnet_input_shapes(self):
        result = {}
        if self.edge_function is not None:
            result.update({"edge_function"  : [i.shape for i in self.edge_function.inputs]})
        result.update({"node_function":[i.shape for i in self.node_function.inputs]})
        return result

    def get_graphnet_output_shapes(self):
        result = {}
        if self.edge_function is not None:
            result.update({"edge_function"  : [i.shape for i in self.edge_function.outputs]})
        result.update({"node_function":[i.shape for i in self.node_function.outputs]})
        return result

    def get_graphnet_output_shapes(self):
        return [i.shape for i in self.edge_function.outputs] , [i.shape for i in self.node_function.outputs]

    def summary(self):
        print("Summary of all GraphNet functions")
        def print_summary_if_keras_model(mod_, which_mod):
            print(which_mod)
            if isinstance(mod_,tf.keras.Model):
                mod_.summary()
            else:
                print(mod_)
            print("")
        
        print_summary_if_keras_model(self.node_function,'Node function')
        print_summary_if_keras_model(self.edge_function,'Edge function')
        print_summary_if_keras_model(self.edge_function,'Edge Agg. function')
        
    def weights(self):
        """
        returns the weights of all the functions associated with this GN in a vector.

        These are: 
          node_function, edge_function, edge_aggregation_function

        """
        all_weights = [ *self.edge_function.weights, *self.node_function.weights]
        if self.node_to_prob_function is not None:
            all_weights.extend(self.node_to_prob_function.weights)
        
        if not self.is_graph_independent:
            if self.edge_aggregation_function is not None and not isinstance(self.edge_aggregation_function, type(tf.reduce_mean)):
            # If the edge aggregation function has weights (it is not a simple aggregation like "mean") accum. the weights
                all_weights.extend(self.edge_aggregation_function.weights)
            
        return all_weights
    
    def observe_nodes(self, graph):
        probs = [];
        for n in graph.nodes:
            probs.append(self.node_to_prob_function(n.node_attr_tensor))
            
        return probs
        
    def observe_node(self, node):
        self.node_to_prob_function(node)

    def __call__(self, graph):
        return self.graph_eval(graph)

    def graph_tuple_eval(self, tf_graph_tuple):
        # This method parallels what the deepmind library does for faster batched computation. 
        # The `tf_graph_tuple` contains edge, nodes, n_edges, n_nodes, senders (indices), receivers (indices).
        # * the "edges" and "nodes" are already stacked into a tensor
        # * If .from_node or .to_node tensors are needed for the edge computations they are gathered according to the senders and receivers tensors.
        # * the edge function is applied to edges
        # * the edge function outputs are aggregated according to the "receivers" tensor to yield the messages.
        # 

        # if the graph is not graph_indep compute the edge-messages with the aggregator.
        # 1) compute the edge functions
        edge_inputs  = {}
        if EdgeInput.EDGE_STATE.value in self.edge_input_dict.keys():
            v = tf_graph_tuple.edges
            edge_inputs.update({EdgeInput.EDGE_STATE.value : v})

        if EdgeInput.SENDER_NODE_STATE.value in self.edge_input_dict.keys():
            v  = tf.gather(tf_graph_tuple.nodes, tf_graph_tuple.senders, axis = 0)
            edge_inputs.update({EdgeInput.SENDER_NODE_STATE.value : v})

        if EdgeInput.RECEIVER_NODE_STATE.value in self.edge_input_dict.keys():
            v = tf.gather(tf_graph_tuple.nodes, tf_graph_tuple.receivers,axis = 0)
            edge_inputs.update({EdgeInput.RECEIVER_NODE_STATE.value : v })

        tf_graph_tuple.edges = self.edge_function(edge_inputs)

        # 2) Aggregate the messages (unsorted segment sums etc):
        if self.has_seg_aggregator and (NodeInput.EDGE_AGG_STATE.value in self.node_input_dict.keys()):
            if NodeInput.EDGE_AGG_STATE.value in self.node_input_dict.keys():
                max_seq = int(tf.reduce_sum(tf_graph_tuple.n_nodes))
                messages = self.edge_aggregation_function_seg(tf_graph_tuple.edges, tf_graph_tuple.receivers, max_seq)

        else:
            Exception("Not Implemented!")
            # This is the more efficient aggregator that makes use of the structure of the GraphTuple indexing.


        # 3) Compute node function:
        node_inputs = {}
        if NodeInput.EDGE_AGG_STATE.value in self.node_input_dict.keys():
            node_inputs.update({NodeInput.EDGE_AGG_STATE.value : messages})
        if NodeInput.NODE_STATE.value in self.node_input_dict.keys():
            node_inputs.update({NodeInput.NODE_STATE.value : tf_graph_tuple.nodes})
        if NodeInput.GLOBAL_STATE.value in self.node_input_dict.keys():
            Exception("Not implemented")
        
        tf_graph_tuple.nodes = self.node_function(node_inputs)
        return tf_graph_tuple
        # no changes in Graph topology - nothing else to do!

    def graph_eval(self, graph, eval_mode = "batched"):
        # Evaluate the full GraphNet step: ("naive" implementations - used for testing)
        # parameters:
        #   eval_mode - ['batched' , 'safe']: 
        #                batched: assuming that the first non-batch dimension of the attributes of the nodes and 
        #                         edges of the input graph correspond to different graphs of the same type (same connectivity)
        #                         this allows for some more parallelization. The input attributes should 
        #                         be at least two dimensional (a 1d vector [Batchdim, N] is entered as [Batchdim, 1, N] tensor.
        # 
        #                safe:    This is a naive implementation. It is useful for testing and providing some memory 
        #                         efficiency when dealing with larger graphs ('batched' method may fail in such cases since 
        #                         it creates big intermediate tensors)
        #                         
        # 
        # 
        eval_mode_dict = {'batched' : self._graph_eval_batched,'safe' : self._graph_eval_safe}
        return eval_mode_dict[eval_mode](graph)

    def _graph_eval_safe(self, graph):
        # 
        #                safe:    This is a naive implementation. It is useful for testing and providing some memory 
        #                         efficiency when dealing with larger graphs ('batched' method may fail in such cases 
        #                         since it creates big intermediate tensors).
        # 
        # 
        assert(isinstance(graph, Graph))
        eval_mode = 'safe'

        self.eval_edge_functions_safe(graph)

        batch_size             = graph.nodes[0].shape[0]; # This will be related to the input graph tuple.

        edge_input_size = self.edge_input_size ; # This relates to the graphnet being evaluated.

        # For GraphIndependent there are actualy no messages to be passed... 
        # I made a vector of zeros to treat 
        edge_to_node_agg_dummy = np.zeros([batch_size, edge_input_size]);

        # Compute the edge-aggregated messages:
        
        
        # Simply looping over nodes: Here either the interm. results of aggregation etc are computed or batches are prepared.
        for n in graph.nodes: # this explicit iteration is expensive and unnecessary in most cases. The DM approach (graph tuples) seems better - implement that.
            if len(n.incoming_edges) != 0:
                if not self.is_graph_independent :
                    #print([e.edge_tensor for e in n.incoming_edges])
                    edge_vals_ = tf.stack([e.edge_tensor for e in n.incoming_edges])
                    #print(edge_vals)
                    print(self.edge_aggregation_function)
                    edge_to_node_agg = self.edge_aggregation_function(edge_vals_)
                else:
                    edge_to_node_agg = edge_to_node_agg_dummy
            else:
                None #edge_to_node_agg = edge_to_node_agg_dummy

            if self.is_graph_independent:
                node_attr_tensor = self.node_function([n.node_attr_tensor])
                n.set_tensor(node_attr_tensor)
            else:
                node_attr_tensor = self.node_function([edge_to_node_agg, n.node_attr_tensor])
                n.set_tensor(node_attr_tensor)

        return graph

    def save(self, path):
        functions = [self.node_function, self.edge_aggregation_function, self.edge_function, self.node_to_prob_function]
        path_labels = ["node_function", "edge_aggregation_function", "edge_function", "node_to_prob"]
        import os
        if not os.path.exists(path):
            os.makedirs(path)
            
        for model_fcn, label in zip(functions, path_labels):
            if model_fcn is not None:
                d_ = os.path.join(path,label)
                model_fcn.save(d_)
                
    @staticmethod
    def load_graph_functions(path):
        """
        Returns a list of loaded graph functions.
        """
        function_rel_paths = ["node_function", "edge_aggregation_function", "edge_function", "node_to_prob"]
        functions = {};

        if not os.path.exists(path):
            print("path does not exist.")
            assert(0)
            
        avail_functions = os.listdir(path) # the path should have appropriately named folders that correspond to the diffferent graph functions. All are keras models.
        for l in function_rel_paths:
            d_ = os.path.join(path,l)
            if not os.path.exists(d_):
                print("path %s does not exist! Function %s will not be constructed."%(d_,l))
                
            else:
                model_fcn = tf.keras.models.load_model(d_)

                functions.update({l:model_fcn})

                print("loading %s"%(d_))
        return functions

    def _graph_eval_batched(self, graph):
        # Evaluate the full GraphNet step:
        # parameters:
        #   eval_mode - ['batched' , 'safe']: 
        #                batched: assuming that the first non-batch dimension of the attributes of the nodes and 
        #                         edges of the input graph correspond to different graphs of the same type (same connectivity)
        #                         this allows for some more parallelization. The input attributes should 
        #                         be at least two dimensional (a 1d vector [Batchdim, N] is entered as [Batchdim, 1, N] tensor.
        # 
        # 
        assert(isinstance(graph, Graph))

        self.eval_edge_functions_batched(graph)

        batch_size             = graph.nodes[0].shape[0]; # This will be related to the input graph tuple.

        edge_input_size = self.edge_input_size ; # This relates to the graphnet being evaluated.

        # For GraphIndependent there are actualy no messages to be passed... 
        # I made a vector of zeros to treat 
        edge_to_node_agg_dummy = np.zeros([batch_size, edge_input_size]);

        # Compute the edge-aggregated messages:
        edge_agg_messages_batch = []
        node_states_batch = []
        
        # Simply looping over nodes: Here either the interm. results of aggregation etc are computed or batches are prepared.
        for n in graph.nodes: # this explicit iteration is expensive and unnecessary in most cases. The DM approach (graph tuples) seems better - do that.
            if NodeInput.EDGE_AGG_STATE.value in self.node_input_dict.keys():

                if len(n.incoming_edges) != 0:
                    edge_vals_ = tf.stack([e.edge_tensor for e in n.incoming_edges])
                    edge_to_node_agg = self.edge_aggregation_function(edge_vals_)
                else: 
                    # Put a vector of "zeros" in place of the aggregation - there are no edges to accum. over!
                    edge_to_node_agg = edge_to_node_agg_dummy 

                edge_agg_messages_batch.append(edge_to_node_agg)

            node_states_batch.append(n.node_attr_tensor)
        

        # out of loop: either accummulated lists for batched computation or evaluated edges and nodes sequentially
        # If we compute in batched mode, there is some reshaping to be done in the end. 
        node_input_shape = graph.nodes[0].shape # nodes and edges (therefore graphs as well) could contain multiple datapoints. This is to treat this case.
        node_output_shape =self.node_function.output.shape


        node_function_inputs = {}
        if NodeInput.NODE_STATE.value in self.node_input_dict.keys():
            node_states_concat = tf.concat(node_states_batch,axis = 0)
            node_function_inputs.update({NodeInput.NODE_STATE.value: node_states_concat})
        if NodeInput.EDGE_AGG_STATE.value in self.node_input_dict.keys():
            edge_agg_messages_concat = tf.concat(edge_agg_messages_batch,axis = 0)
            node_function_inputs.update({NodeInput.EDGE_AGG_STATE.value : edge_agg_messages_concat})
        if NodeInput.GLOBAL_STATE.value in self.node_input_dict.keys():
            Exception("Globals not implemented yet.")
            node_function_inputs.update({NodeInput.GLOBAL_STATE.value : global_value_concat})

        batch_res = self.node_function(node_function_inputs)

        unstacked = tf.unstack(
                tf.reshape(
                    batch_res,[-1,*node_input_shape[0:1],*node_output_shape[1:]]), axis = 0
                )

        for n, nvalue in zip(graph.nodes, unstacked):
            n.set_tensor(nvalue)

        return graph

    def save(self, path):
        functions = [self.node_function, self.edge_aggregation_function, self.edge_function, self.node_to_prob_function]
        path_labels = ["node_function", "edge_aggregation_function", "edge_function", "node_to_prob"]
        import os
        if not os.path.exists(path):
            os.makedirs(path)
            
        for model_fcn, label in zip(functions, path_labels):
            if model_fcn is not None:
                d_ = os.path.join(path,label)
                model_fcn.save(d_)
                
    @staticmethod
    def load_graph_functions(path):
        """
        Returns a list of loaded graph functions.
        """
        function_rel_paths = ["node_function", "edge_aggregation_function", "edge_function", "node_to_prob"]
        functions = {};

        if not os.path.exists(path):
            print("path does not exist.")
            assert(0)
            
        avail_functions = os.listdir(path) # the path should have appropriately named folders that correspond to the diffferent graph functions. All are keras models.
        for l in function_rel_paths:
            d_ = os.path.join(path,l)
            if not os.path.exists(d_):
                print("path %s does not exist! Function %s will not be constructed."%(d_,l))
                
            else:
                model_fcn = tf.keras.models.load_model(d_)

                functions.update({l:model_fcn})

                print("loading %s"%(d_))
        return functions

    def save(self, path):
        functions = [self.node_function, self.edge_aggregation_function, self.edge_function, self.node_to_prob_function]
        path_labels = ["node_function", "edge_aggregation_function", "edge_function", "node_to_prob"]
        import os
        if not os.path.exists(path):
            os.makedirs(path)
            
        for model_fcn, label in zip(functions, path_labels):
            if model_fcn is not None:
                d_ = os.path.join(path,label)
                model_fcn.save(d_)
                
    @staticmethod
    def load_graph_functions(path):
        """
        Returns a list of loaded graph functions.
        """
        function_rel_paths = ["node_function", "edge_aggregation_function", "edge_function", "node_to_prob"]
        functions = {};

        if not os.path.exists(path):
            print("path does not exist.")
            assert(0)
            
        avail_functions = os.listdir(path) # the path should have appropriately named folders that correspond to the diffferent graph functions. All are keras models.
        for l in function_rel_paths:
            d_ = os.path.join(path,l)
            if not os.path.exists(d_):
                print("path %s does not exist! Function %s will not be constructed."%(d_,l))
                
            else:
                model_fcn = tf.keras.models.load_model(d_)

                functions.update({l:model_fcn})

                print("loading %s"%(d_))
        return functions

    def load(self, path):
        """
        Load a model from disk. If the model is already initialized the current graphnet functions are simply overwritten. 
        If the model is un-initialized, this is called from a static method (factory method) to make a new object with consistent properties.

        """
        functions = [self.node_function, self.edge_aggregation_function, self.edge_function, self.node_to_prob_function]
        all_paths = ["node_function", "edge_aggregation_function", "edge_function", "node_to_prob"]
        path_label_to_function = {z:v for z,v in zip(all_paths,functions)}
        path_labels = os.listdir(path) #
        
        if not os.path.exists(path):
            print("path does not exist.")
            assert(0)
            
        for l in path_labels:
            d_ = os.path.join(path,l)
            if not os.path.exists(d_):
                print("path %s does not exist! Function %s will not be constructed."%(d_,l))
                next
            else:
                model_fcn = tf.keras.models.load_model(d_)
                path_label_to_function[l] = model_fcn
                print(path_label_to_function[l] )

                print("loading %s"%(d_))

    def eval_edge_functions_safe(self,graph):
        """
        Evaluate all edge functions. Batched mode has some shape-juggling going on.
        If you see weird behaviour that's the first place to look for - test should fail if this does not compute properly.
        
        params:
          graph     - the graph containing the edges 
        """
        if len(graph.edges) == 0:
            return 

        def prep_inputs(edge_):
            edge_inputs = {};
            if EdgeInput.EDGE_STATE.value in self.edge_input_dict.keys():
                edge_inputs.update({EdgeInput.EDGE_STATE.value : edge_.edge_tensor})

            if EdgeInput.SENDER_NODE_STATE.value in self.edge_input_dict.keys():
                edge_inputs.update({EdgeInput.SENDER_NODE_STATE.value : edge_.node_from.node_attr_tensor})

            if EdgeInput.RECEIVER_NODE_STATE.value in self.edge_input_dict.keys():
                edge_inputs.update({EdgeInput.RECEIVER_NODE_STATE.value : edge_.node_to.node_attr_tensor})
            return edge_inputs
        
        for edge in graph.edges:
            edge_tensor = self.edge_function( prep_inputs(edge) )
            edge.set_tensor(edge_tensor)
                    

            
    def eval_edge_functions_batched(self,graph):
        """
        Evaluate all edge functions. Batched mode has some shape-juggling going on.
        If you see weird behaviour that's the first place to look for - test should fail if this does not compute properly.
        
        params:
          graph     - the graph containing the edges 
          eval_mode - "safe" or "batched" (batched is also safe if state shapes are respected)
        """
        
        if len(graph.edges) == 0:
            return 
        
        edges_ = graph.edges 
        edges_shape = edges_[0].shape # not good! I could have edges of different types/sizes. Not sure how to deal with this at the moment.
        edge_inputs = {}

        if EdgeInput.EDGE_STATE.value in self.edge_input_dict.keys():
            batched_edge_states = tf.concat([e.edge_tensor for e in edges_],axis = 0)
            edge_inputs.update({EdgeInput.EDGE_STATE.value : batched_edge_states})


        if EdgeInput.SENDER_NODE_STATE.value in self.edge_input_dict.keys():
            node_from_concat = tf.concat([e.node_from.node_attr_tensor for e in edges_], axis = 0)
            edge_inputs.update({EdgeInput.SENDER_NODE_STATE.value : node_from_concat})

        if EdgeInput.RECEIVER_NODE_STATE.value in self.edge_input_dict.keys():
            node_to_concat= tf.concat([e.node_to.node_attr_tensor for e in edges_],axis = 0)
            edge_inputs.update({EdgeInput.RECEIVER_NODE_STATE.value : node_to_concat})

        batch_res = self.edge_function(edge_inputs)

        unstacked = tf.unstack(tf.transpose(tf.reshape(batch_res,[-1,*edges_shape[0:1],*batch_res.shape[1:]]),[0,1,2]), axis = 0)

        for e, evalue in zip(edges_, unstacked):
            e.set_tensor(evalue)

                
def make_mlp(units, input_tensor_list , output_shape):
    """
    A default method for making a small MLP:
    """
    if len(input_tensor_list) > 1:
        edge_function_input = keras.layers.concatenate(input_tensor_list);
    else:
        edge_function_input = input_tensor_list[0] #essentialy a list of a single tensor.
    #print(units, edge_function_input)

    y = Dense(units)(  edge_function_input)
    y=  Dropout(rate = 0.2)(y)
    y = Dense(units, activation = "relu")(y)
    y=  Dropout(rate = 0.2)(y)
    y = Dense(units, activation = "relu")(y)
    y = Dense(output_shape[0])(y)
    return tf.keras.Model(inputs = input_tensor_list, outputs = y)


def make_node_mlp(units,
        edge_state_input_shape = None,
        node_state_input_shape = None, 
        node_emb_size= None,
        use_global_input = False,
        use_edge_state_agg_input = True,
        graph_indep = False):

    if use_edge_state_agg_input and graph_indep:
        Exception("requested a GraphIndep graphnet node function but speciffied use of node input! This is inconsistent.")

    node_state_in = Input(shape = node_state_input_shape, name = NodeInput.NODE_STATE.value);

    if graph_indep:
        return make_mlp(units, [node_state_in], node_emb_size)

    if use_edge_state_agg_input:
        agg_edge_state_in = Input(shape = edge_state_input_shape, name =  NodeInput.EDGE_AGG_STATE.value);
        return make_mlp(units, [agg_edge_state_in, node_state_in],node_emb_size)


def make_edge_mlp(units,
        edge_state_input_shape = None,
        edge_state_output_shape = None, 
        sender_node_state_output_shape = None,
        receiver_node_state_shape = None,
        edge_output_state_size = None,
        use_edge_state = True,
        use_sender_out_state = True,
        use_receiver_state = True,
        graph_indep = False):
    """
    When this is a graph independent edge function, the input is different: 
    The node states from the sender and receiver are not used!
    """
    tensor_in_list = []
    if use_edge_state:
        edge_state_in = Input(
                shape = edge_state_input_shape, name =EdgeInput.EDGE_STATE.value) 
        tensor_in_list.append(edge_state_in)

    if use_sender_out_state:
        node_state_sender_out = Input(
                shape = sender_node_state_output_shape, name = EdgeInput.SENDER_NODE_STATE.value);
        tensor_in_list.append(node_state_sender_out)

    if use_receiver_state:
        node_state_receiver_in = Input(
                shape = receiver_node_state_shape, name = EdgeInput.RECEIVER_NODE_STATE.value);
        tensor_in_list.append(node_state_receiver_in)

    if graph_indep:
        try:
            assert(use_sender_out_state is False)
            assert(use_receiver_out_state is False)
        except:
            ValueError("The receiver and sender nodes for graph-independent blocks should not be used! It was attempted to create an edge function for a graph-indep. block containing receiver and sender states as inputs.")
        ## Building the edge MLP:
        tensor_input_list = [edge_state_in]
        return make_mlp(units,tensor_input_list,edge_output_state_size )  
    else:
        ## Building the edge MLP:
        return make_mlp(units,tensor_in_list,edge_output_state_size )  

def make_keras_simple_agg(input_size, agg_type):
    """
    For consistency I'm making this a keras model (easier saving/loading)
    This is for the "naive" graphNet evaluators. There is a fully batched 
    aggregator with segment sums that should be preferred.

    parameters:
       input_size : the size of the expected input - should fail (even later at some point)
       agg_type   : ['mean','sum','min','max']
    """
    dict_agg = {
            'mean' : (tf.reduce_mean,tf.math.unsorted_segment_mean),
            'sum'  : (tf.reduce_sum,tf.math.unsorted_segment_sum),
            'max'  : (tf.reduce_max,tf.math.unsorted_segment_max),
            'min'  : (tf.reduce_min,tf.math.unsorted_segment_min)}

    x = Input(shape = input_size, name = "edge_messages") # for "segment" aggregators this needs also a bunch of indices!
    aggs = dict_agg[agg_type]
    y = aggs[0](x,0)
    m_basic = tf.keras.Model(inputs = x, outputs = y,name = 'basic_%s_aggregator'%agg_type)

    x2 = Input(shape = input_size, name = "edge_messages") # for "segment" aggregators this needs also a bunch of indices!
    x2_inds = Input(shape = (1,), dtype = "int32", name = "receiver_indices") # for the segment indices.
    x2_ms = Input(shape = (1,), batch_size = 1 ,dtype = "int32",name = "max_segment_idx")

    print(input_size)
    #y2 = dict_agg[agg_type][1](x2,x2_inds,x2_ms)
    m_seg = dict_agg[agg_type][1] #tf.keras.Model(inputs = [x2,x2_inds,x2_ms] , outputs = y2,name= 'segm_%s_aggregator'%agg_type)
    
    return m_basic, m_seg

def make_pna_unsc(input_shape):
    """
    unscaled principal neighborhood aggregator
    Corso, Gabriele, et al. "Principal neighbourhood aggregation for graph nets." arXiv preprint arXiv:2004.05718 (2020).
    """
    print("Not implemented yet! - though simple - apply mean/max/min/sum and linearly combine (maybe a LayerNorm fits well on the output.")
    assert(0)
    return None

def edge_aggregation_function_factory(input_shape, agg_type = 'mean'):
    """
    A factory method to create aggregators for graphNets. 
    Somehow I will have to adapt this for unsorted segment reductions (like DeepMind does) because it's much faster.
    Implemented aggregators:
    ----------------------
      sum
      mean
      max
      min

    todo:
    ---------------------
      pna_unsc
      std
      moment
      
    """
    # Each lambda creates a "basic" and a "segment" (sparse) aggregator.
    agg_type_dict = {
            'mean' : lambda input_shape : make_keras_simple_agg(input_shape, 'mean'),
            'sum' : lambda input_shape : make_keras_simple_agg(input_shape, 'sum'),
            'max' : lambda input_shape : make_keras_simple_agg(input_shape, 'max'),
            'min' : lambda input_shape : make_keras_simple_agg(input_shape, 'min'),
            'pna_unsc' : make_pna_unsc # PNA aggregator without degree scalers
            }

    aggregators = agg_type_dict[agg_type](input_shape)
    return aggregators


def make_mlp_graphnet_functions(units,
        node_input_size, node_output_size, 
        edge_input_size = None, edge_output_size = None,
        graph_indep = False, message_size = None, aggregation_function = 'mean'):
    """
    Make the 3 functions that define the graph (no Nodes to Global and Edges to Global)
    * If `edge_input_size' and `edge_output_size' are not defined (None) 
      It is assumed that all inputs and all outputs are the same shape for nodes and edges.

    parameters:
      node_input_size  : shape of the node state of the incoming graph.
      node_output_size : shape of the node state of the output graph
      edge_input_size  : (optional) the shape of the edge attributes of the input graph (defaults to node_input_size)
      edge_output_size : (optional) the shape of the edge attributes of the output graph (defaults to node_output_size)
      graph_indep      : default: False - whether message passing happens in this graphNet (if not it is a "graph-independent" GN)
      message_size     : (optional) the size of the passed message - this can be different from the edge-state output size! 
                         What this affects is what goes into the aggregator and to the node-function. Edge state can still be whatever.


    """
    node_input = node_input_size
    node_output = node_output_size
    if edge_input_size is None:
        edge_input_size = node_input_size
    if edge_output_size is None:
        edge_output_size = node_output_size

    if message_size is None:
        message_size = edge_output_size

    edge_mlp_args = {"edge_state_input_shape" : (edge_input_size,),
            "edge_state_output_shape" : (message_size,),
            "sender_node_state_output_shape" : (node_input,), # this has to be compatible with the output of the aggregator.
            "edge_output_state_size":(edge_output_size,),
            "receiver_node_state_shape" : (node_input,)} 

    edge_mlp_args.update({"graph_indep" : graph_indep})
    edge_mlp = make_edge_mlp(units, **edge_mlp_args)

    if not graph_indep:
        input_shape = [None, *edge_mlp.outputs[0].shape[1:]]
        agg_fcn  = edge_aggregation_function_factory(input_shape, aggregation_function) # First dimension - incoming edge index
    else:
        #edge_mlp = None
        agg_fcn = None

    node_mlp_args = {"edge_state_input_shape": (edge_input_size,),
            "node_state_input_shape" : (node_input,),
            "node_emb_size" : (node_output_size,)}
    node_mlp_args.update({"graph_indep" : graph_indep})

    node_mlp = make_node_mlp(units, **node_mlp_args)
    return {"edge_function" : edge_mlp, "node_function": node_mlp, "edge_aggregation_function": agg_fcn} # Can be passed directly  to the GraphNet construction with **kwargs


