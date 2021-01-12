""" Classes for basic manipulation of GraphNet """
import numpy as np
import tensorflow as tf

def _copy_any_ds(val):
    """
    Copy semantics for different datatypes accepted.
    This affects what happens when copying nodes, edges and graphs. 
    In order to trace gradients, 
    and defines a consistent interface regardless of the input data-structure.
    """
    valout = val
    if isinstance(val , np.ndarray) or isinstance(val, list):
        valout = val.copy()

    if isinstance(val, tf.Variable) or isinstance(val,tf.Tensor):
        valout = tf.identity(val) # TODO: maybe have a flag to override this? Adding more ops does not always make sense.
    
    return valout

class Node:
    def __init__(self, node_attr_tensor):
        if len(node_attr_tensor.shape) <2:
            raise ValueError("The shape of the input for nodes and edges should have at least 2 dimensions!")
        self.node_attr_tensor = node_attr_tensor
        self.incoming_edges = [];
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
        return Node(self.node_attr_tensor  - n.node_attr_tensor)
    
class Edge:
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
        node_to   = nodes_correspondence[self.node_to]
        return Edge(edge_tensor, node_from, node_to)

    def __add__(self, edge):
        Exception("Edge addition is not implemented! This is due to potentially unclear semantics. Perform this manually.")


class Graph:
    def __init__(self, nodes, edges, NO_VALIDATION=True):
        """
        Creates a graph from a set of edges and nodes
        """
        self.nodes = nodes
        self.edges = edges
        if not NO_VALIDATION:
            self.validate_graph()


    def is_equal_by_value(self,g2):
        """
        Checks if the graphs have the same values for node and edge attributes
        """
        is_equal = True
        for n1,n2 in zip(self.nodes, g2.nodes):
            is_equal = is_equal and tf.reduce_all(n1.node_attr_tensor == n2.node_attr_tensor)

        for e1, e2 in zip(self.edges, g2.edges):
            is_equal = is_equal and tf.reduce_all(e1.edge_tensor== e2.edge_tensor)
        
        return bool(is_equal)
    
    def compare_connectivity(self,g2):
        """
        Checks if the connectivity of two graphs is the same.
        """
        g1 = self
        nodes_from_match = [(g1.nodes.index(e1.node_from) == g2.nodes.index(e2.node_from)) for e1,e2 in zip(g1.edges,g2.edges)]
        nodes_to_match = [(g1.nodes.index(e1.node_to) == g2.nodes.index(e2.node_to)) for e1,e2 in zip(g1.edges,g2.edges)]
        all_matching = True
        for matches in [*nodes_from_match, *nodes_to_match]:
            all_matching = all_matching and matches
        return all_matching



    @staticmethod
    def validate_graph(self):

        # validate that the edges are all 
        for e in self.edges:

            if ((e.node_from in self.nodes)):
                raise AssertionError("The source node {nn} for edge {ee} is not in the graph!".format(nn = e.node_from, ee = e))
            if (e.node_to in self.nodes):
                raise AssertionError("The destination node {nn} for edge {ee} is not in the graph!".format(nn = e.node_to, ee = e))


    def copy(self):
        # copy attributes of nodes and edges and re-create graph connectivity:
        nodes_coppied = [n.copy() for n in self.nodes]
        nodes_correspondence = {s:c for s , c in zip(self.nodes,nodes_coppied)}
        # Instantiate the new edges:
        coppied_edge_instances = []
        for e in self.edges:
            enew = e.copy(nodes_correspondence) 
            coppied_edge_instances.append(enew)
        return Graph(nodes_coppied, coppied_edge_instances)


    def get_subgraph_from_nodes(self, nodes, edge_trimming_mode = "+from+to"):
        """
        Node should belong to graph. Creates a new graph with coppied edge and
        node properties, defined from a sub-graph of the original graph.
        parameters:
          self (type = Graph): the graph we want a sub-graph from
          nodes: the nodes of the graph we want the subgraph of.
          mode:  "+from+to" - keep an edge if there is a "from" node or a "to" node at that edge (and the corresponding node)
                 "-from-to" - keep an edge if there is NOT a "from" node and NOT a "to" node at that edge (and the corresponding node)
                 "+from"    - keep an edge only if it has a "from" node that coincides with any of the nodes in the list (not implemented)
                 "+to"      - keep an edge only if it has a "to" node that coincides with any of the nodes in the list (not implemented)
                 "-from"    - keep an edge only if it DOESN't have a "from" node that concides with any of the nodes in the list (not implemented)
        """

        def check_edge_trimming_condition(e_):
            if edge_trimming_mode == "+from+to":
                return (e.node_from in nodes) and (e.node_to in nodes)

            if edge_trimming_mode == "-from+to":
                return (e.node_from not in nodes) and (e.node_to not in nodes)
                

        sg_nodes_copy = [n.copy() for n in nodes]
        original_copy_nodes_correspondence = {n:nc for n, nc in zip(nodes, sg_nodes_copy)}
        sg_edges_copy = [];
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
        nodes = [nself + n for nself,n in zip(self.nodes,graph.nodes)]
        correspondence = {s:t for s, t in zip(self.nodes,nodes)}
        added_edges = [];
        for eself,e in zip(self.edges, graph.edges):
            enew = Edge(eself.edge_tensor +  e.edge_tensor, 
                    correspondence[eself.node_from], 
                    correspondence[eself.node_to])
            added_edges.append(enew);

        return Graph(nodes, added_edges)

def make_graph_tuple_from_graph_list(list_of_graphs):
    """
    Takes in a list of graphs (with consistent sizes - not checked)
    and creates a graph tuple (input tensors + some book keeping)
    
    Because there is some initial functionality I don't want to throw away currently, that implements special treatment for nodes and edges
    coming from graphs with the same topology, it is currently required that the first dimension of nodes and edges
    for the list of graphs that are entered in this function is always 1 (this dimension is the batch dimension in the previous implementation.)
    """
    
    # check the first dimension is 1 - instruct to split graphs if not.
    problematic_graphs = []
    
    # TODO: Support splitting a list of same graphs with the first dimension of node and edge 
    #       features different than one and constructing a GraphTuple. Currently the first 
    #       dimension is required to be "1" (but squeezed later on!)
    for g_index,g in enumerate(list_of_graphs):
        problem = ''
        all_sizes_same = True
        if g.nodes[0].get_state().shape[0] != 1:
            problem += 'First size of node attributes should be 1 - found %i '%g.edges[0].get_state().shape[0]
        if g.edges[0].edge_tensor.shape[0] != 1:
            problem += 'First size of node attributes should be 1 - found %i '%g.edges[0].get_state().shape[0]
            
    # graph_id = [id_ for id_, dummy in enumerate(list_of_graphs)]
    all_edges, all_nodes, n_nodes,n_edges =[[],[],[],[]]
    for g in list_of_graphs:
        all_edges.extend(g.edges)
        all_nodes.extend(g.nodes)
        n_nodes.append(len(g.nodes)) 
        n_edges.append(len(g.edges)) 
    
    edge_attr_tensor, nodes_attr_tensor, senders, receivers = [[],[],[],[]];
    for e in all_edges:
        edge_attr_tensor.append(e.edge_tensor)
        senders.append(all_nodes.index(e.node_from))
        receivers.append(all_nodes.index(e.node_to))
        
    
    for n in all_nodes:
        nodes_attr_tensor.append(n.node_attr_tensor)
    
    # The 2nd dimension (dimension index 1) should be of size 1 (there is a test in the start of the constructor).
    # The same framework supports efficient computation on graphs of the same topology batched together where the first dimension 
    # is the batched size. It is required that such graphs were provided for the construction (or at least the first dimension is "1").
    edges_attr_stacked = tf.squeeze(tf.stack(edge_attr_tensor,0),1) 
    nodes_attr_stacked = tf.squeeze(tf.stack(nodes_attr_tensor,0),1)
    return GraphTuple(nodes_attr_stacked, edges_attr_stacked,senders, receivers, n_nodes, n_edges)# , graph_id)


class GraphTuple:
    def __init__(self, nodes, edges,senders,receivers, n_nodes, n_edges, sort_receivers_to_edges  = False):
        """
        A graph tuple contains multiple graphs for faster batched computation. 
        
        parameters:
            nodes      : a `tf.Tensor` containing all the node attributes
            edges      : a `tf.Tensor` containing all the edge attributes
            senders    : a list of sender node indices defining the graph connectivity. The indices are unique accross graphs
            receivers  : a list of receiver node indices defining the graph connectivity. The indices are unique accross graphs
            n_nodes    : a list, a numpy array or a tf.Tensor containing how many nodes are in each graph represented by the nodes and edges in the object
            n_edges    : a list,a numpy array or a tf.Tensor containing how many edges are in each graph represented by the nodes and edges in the object
            sort_receivers :  whether to sort the edges on construction, allowing for not needing to sort the output of the node receiver aggregators.
        """
        # Sort edges according to receivers and sort receivers:
        assert(len(n_nodes) == len(n_edges))
        
        self.nodes = nodes # floats tensor
        self.edges = edges # floats tensor
        self.senders = senders     # integers
        self.receivers = receivers # integers
        self.n_nodes = n_nodes     # integers
        self.n_edges = n_edges     # integers
        self.n_graphs = len(self.n_nodes) # assuming the n_nodes is a list containing the number of nodes for each graph.

        graph_indices_nodes = []
        for k_,k in enumerate(self.n_nodes):
            graph_indices_nodes.extend(np.ones(k).astype("int")*k_)

        graph_indices_edges = []
        for k_,k in enumerate(self.n_edges):
            graph_indices_edges.extend(np.ones(k).astype("int")*k_)

        self.graph_indices_nodes , self.graph_indices_edges = graph_indices_nodes, graph_indices_edges
    

    def is_equal_by_value(self, other_graph_tuple):
        v1 = self.edges,self.nodes, self.receivers,self.senders, self.n_nodes, self.n_edges, self.n_graphs
        v2 = other_graph_tuple.edges,other_graph_tuple.nodes, other_graph_tuple.receivers,other_graph_tuple.senders, other_graph_tuple.n_nodes, other_graph_tuple.n_edges, other_graph_tuple.n_graphs
        def _equals_or_all_equals(v1_,v2_):
            if isinstance(v1_, list) and isinstance(v2_, list):
                return v1_ == v2_
            if isinstance(v1_, tf.Variable) and isinstance(v2_, tf.Variable):
                return all(v1_ == v2_)
            if isinstance(v1_, np.array) and isinstance(v2_. np.array):
                return all(v1_ == v2_)
        return all([_equals_or_all_equals(v1__,v2__) for v1__, v2__ in zip(v1,v2)])

    def copy(self):
        n = _copy_any_ds(self.nodes)
        e = _copy_any_ds(self.edges)
        s = _copy_any_ds(self.senders)
        r = _copy_any_ds(self.receivers)
        nnodes = _copy_any_ds(self.n_nodes)
        nedges = _copy_any_ds(self.n_edges)
        ngraphs = _copy_any_ds(self.n_graphs)
        return GraphTuple(n,e,s,r,nnodes,nedges)


    def __add__(self, g2):
        nodes = self.nodes + g2.nodes
        edges = self.edges + g2.edges
        s = self.senders
        r = self.receivers
        n_nodes = self.n_nodes
        n_edges = g2.n_edges
        return GraphTuple(nodes,edges,s,r,n_nodes, n_edges)

        
    def get_graph(self, graph_index):
        """
        Returns a new graph with the same properties as the original  graph.
        gradients are not traced through this operation.
        """
        assert(graph_index >=0 )
        if graph_index > self.n_graphs:
            raise ValueError("The provided index is larger than the available graphs in this GraphTuple object.")
            
        get_start_stop_index = lambda sizes_list, index : np.cumsum([0,*sizes_list[0:index+1]])[-2:]
        start_idx_nodes , end_idx_nodes = get_start_stop_index(self.n_nodes, graph_index)
        start_idx_edges , end_idx_edges = get_start_stop_index(self.n_edges, graph_index)
        nodes_attrs = self.nodes[start_idx_nodes:end_idx_nodes]
        senders, receivers, edge_attr = [v[start_idx_edges:end_idx_edges] for v in [self.senders, self.receivers,self.edges]]
        senders = senders-start_idx_nodes
        receivers = receivers - start_idx_nodes
        nodes = [Node(node_attr[tf.newaxis]) for node_attr in nodes_attrs]
        edges = [Edge(edge_attr_tensor[tf.newaxis], nodes[node_from_idx], nodes[node_to_idx]) for edge_attr_tensor, node_from_idx, node_to_idx in zip(edge_attr, senders,receivers)]
        return Graph(nodes, edges)

        

