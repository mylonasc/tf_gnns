# Tensorflow message passing networks library
A library for easy construction of message-passing networks in tensorflow keras.

# Usage
## Basic data-structures
An attributed multi-graph is a data-structure that contains edges E and nodes (or vertices) V with arbitrary properties (vectors/tensors) embedded in each of these structures.
See the example bellow for the creation of `Edge`, `Node` and `Graph` objects:
```python
# Creation of edges and nodes:
import numpy as np
from ibk_gnns import Graph, Edge, Node

nw_state_size = 10 # the state of the node and edge attributes. They can be different if necessary.

# Defining the connectivity of a graph:
adj_A = [(2,4),(3,4),(2,2) , (2,5),(5,1),(1,2),(2,3),(3,4),(4,5),(6,5),(7,6),(8,7)]

nodes_A = [Node(np.random.randn(1,nw_state_size)) for n in range(10)]
edges_A = [Edge(np.random.randn(1,nw_state_size), node_from= nodes_A[e_ij[0]], node_to= nodes_A[e_ij[1]]) for e_ij in adj_A]
graph_A = Graph(nodes=nodes_A, edges= edges_A)

```

### The `GraphTuple` data structure
The `GraphTuple` is a set of graphs packed into an object that allows for easier parallelization of the GraphNet computation block. 
It is the default (and only) possible data-structure in DeepMind's GraphNets library. In the following the construction of a `GraphTuple`
is demonstrated:

```python
# Some graphs to compute with:
adj_A = [(2,4),(3,4),(2,2) , (2,5),(5,1),(1,2),(2,3),(3,4),(4,5),(6,5),(7,6),(8,7)]
adj_B = [(3,4),(2,2) , (2,5),(5,1),(1,2),(2,3),(3,4),(4,5),(6,5),(7,6),(8,9)]

nw_state_size  = 10

nodes_A = [Node(np.random.randn(1,nw_state_size)) for n in range(10)]
edges_A = [Edge(np.random.randn(1,nw_state_size), node_from= nodes_A[e_ij[0]], node_to= nodes_A[e_ij[1]]) for e_ij in adj_A]
graph_A = Graph(nodes=nodes_A, edges= edges_A)
                
nodes_B = [Node(np.random.randn(1,nw_state_size)) for n in range(10)]
edges_B = [Edge(np.random.randn(1,nw_state_size), node_from=nodes_B[e_ij[0]], node_to = nodes_B[e_ij[1]]) for e_ij in adj_B]
graph_B = Graph(nodes=nodes_B, edges= edges_B)
gt = make_graph_tuple_from_graph_list([graph_A, graph_B])
```

You can retrieve copies of the `Graph` objects by `gt.get_graph(n)` where `n` is the graph you would like to retrieve.

## Creating a custom `GraphNet`
In order to create a GraphNet (without global variables) one needs to define the following:
1. node function
2. an edge function
3. an edge aggregation function (except if a `GraphIndependent` network is implemented)

In addition to that, one needs to pay attention that the input sizes of the node and edge function are consistent with the input graph, the edge aggregation function (if it exists) has to have outputs consistent with expected inputs of the node function (if the graph is not graph independet). Moreover, each of these functions have potentially different inputs related to the input graph. For instance:

* Edge functions may have as inputs (1) the edge state, (2) the sender node state, (3) the receiver node state
* Node functions may have as inputs (1) the node state (2) an aggregated message incoming from the edges that point to that node.
These cases are identified internally by the naming of the inputs of the provided functions. The inputs for edges and nodes can (should) have the following names:

```python

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
  
```

Since one needs consistency between the different functions involved, it's better that these are packed in a factory method.
Here is an example of using such factory methods to create the necessary functions for the popular encode-core-decode GN architecture (popularized by the [DeepMind paper](https://arxiv.org/abs/1806.01261)).

```python

from ibk_gnns import make_mlp_graphnet_functions, GraphNet, Node, Edge, Graph, GraphTuple, make_graph_tuple_from_graph_list

activation = "relu"
gnn_size = 50;
node_input_size_enc, node_output_size_enc = [2, 10]
node_input_size_core, node_output_size_core = [10, 10]
node_input_size_dec, node_output_size_dec = [10,2]
activation = "relu"
edge_input_size_enc  = edge_output_size_dec = 4

graph_fcn_enc = make_mlp_graphnet_functions(gnn_size,
                                          node_input_size = node_input_size_enc,
                                          node_output_size = node_output_size_enc,
                                          edge_input_size =  edge_input_size_enc,
                                          graph_indep=True)

graph_fcn_core = make_mlp_graphnet_functions(gnn_size,
                                          node_input_size = node_input_size_core, 
                                          node_output_size = node_output_size_core, 
                                          graph_indep=False)

graph_fcn_dec = make_mlp_graphnet_functions(gnn_size,
                                          node_input_size = node_input_size_dec, 
                                          node_output_size = node_output_size_dec, 
                                          edge_output_size= edge_output_size_dec,
                                          graph_indep=True)

gnns = [GraphNet(**fcns) for fcns in [graph_fcn_enc,graph_fcn_core,graph_fcn_dec]] # A full encode-core-decode set of GNNs!
    
```

In the example above `make_mlp_graphnet_functions` is the factory method. It returns a set of functions (keras models) that are in turn used to create GNNs.

Each of the `GraphNets` can be easilly saved using the `.save` method. 
You can invoke the summary method of any of these GNNs to inspect the inputs and outputs of the GN functions:
```python
gnns[1].summary()
```
outputs:

```
Summary of all GraphNet functions
Node function
Model: "model_4"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
edge_state_agg (InputLayer)     [(None, 10)]         0                                            
__________________________________________________________________________________________________
node_state (InputLayer)         [(None, 10)]         0                                            
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 20)           0           edge_state_agg[0][0]             
                                                                 node_state[0][0]                 
__________________________________________________________________________________________________
dense_12 (Dense)                (None, 50)           1050        concatenate_1[0][0]              
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, 50)           0           dense_12[0][0]                   
__________________________________________________________________________________________________
dense_13 (Dense)                (None, 50)           2550        dropout_6[0][0]                  
__________________________________________________________________________________________________
dropout_7 (Dropout)             (None, 50)           0           dense_13[0][0]                   
__________________________________________________________________________________________________
dense_14 (Dense)                (None, 50)           2550        dropout_7[0][0]                  
__________________________________________________________________________________________________
dense_15 (Dense)                (None, 10)           510         dense_14[0][0]                   
==================================================================================================
Total params: 6,660
Trainable params: 6,660
Non-trainable params: 0
_____________________________
.
.
.
.
```


For convenience, and to make the syntax a bit more expressive, 
some operators are overloaded. For instance, the sum operator for `GraphTuples` is overloaded. It is possible to compute residual connections
easilly as follows:

```python
def eval_full(G, core_steps =CORE_STEPS):
    ## The actual computation
    gi_str.graph_tuple_eval(G) # happens in-place.

    for ncore in range(core_steps):
        G += gcore.graph_tuple_eval(G.copy())
        
    emb, node_output = node_regressor(G.nodes) # a function operating only on the nodes.
    G.nodes = emb 
    gcore_out = gt_to_global(G) # A function containing node-to-global and edge-to-global aggregators.
    
    return gcore_out, node_output 

```

Currently global blocks are not implemented.
The reason for this is that it is easy to implement them in a few lines when needed.
Here is an example of implementing a graph-to-global function with the respective aggregators:

```python
def make_graph_tuple_to_global(insize = GN_STATE, agg_type = 'mean', global_state_out = 3, type_ = "node", local_bnn = LOCALBNN):
    
    # It would have been much cleaner if I supported this in the library... 
    agg_fcn = make_keras_simple_agg(insize,agg_type) # from ibk_gnns import make_keras_simple_agg
    agg_fcn = agg_fcn[1]
    
    # Constructing the node+edge -> global function. 
    xx = Input(shape = (insize,))
    
    if local_bnn == False:
        out = Dense(insize, 'relu')(xx)
        out = Dense(insize, 'relu')(out)
    else:
        out = Dense(insize)(xx)
        out = tfp.layers.DenseLocalReparameterization(insize,'relu')(out) 
        out = Dense(insize,'relu')(out)

    out = Dense(global_state_out, activation = GLOBAL_OUTPUT_ACTIVATION, use_bias = False)(out)
    
    global_fcn = Model(inputs = xx, outputs = out)
    bnnlosses = global_fcn.losses
    
    def fcn_node_and_edge(gt):
        graph_indices_nodes = []
        for k_,k in enumerate(gt.n_nodes):
            graph_indices_nodes.extend(np.ones(k).astype("int")*k_)

        graph_indices_edges = []
        for k_,k in enumerate(gt.n_edges):
            graph_indices_edges.extend(np.ones(k).astype("int")*k_)
            
        o1 = agg_fcn(gt.nodes,graph_indices_nodes, gt.n_graphs) # node_to_global aggregation
        o2 = agg_fcn(gt.edges,graph_indices_edges, gt.n_graphs) # edge_to_global aggregation
        return global_fcn(o1+o2) # either concat or add the aggregated information.

    def fcn_node(gt):
        graph_indices_nodes = []
        for k_,k in enumerate(gt.n_nodes):
            graph_indices_nodes.extend(np.ones(k).astype("int")*k_)
            
        o1 = agg_fcn(gt.nodes,graph_indices_nodes, gt.n_graphs)
        return global_fcn(o1)

    fcn_dict = {'node': fcn_node,'node_and_edge' : fcn_node_and_edge}
    return fcn_dict[type_], global_fcn, bnnlosses

```




