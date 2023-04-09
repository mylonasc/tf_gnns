# 
# Contains functions for performing some usual operations on tensor dictionaries that correspond
# to graph tuples. 
# 
# 

import tensorflow as tf

GRAPH_TUPLE_STRUCTURE = ['senders', 'receivers', 'n_nodes', 'n_edges', 'global_reps_for_edges', 'global_reps_for_nodes', 'n_graphs']

def _zero_graph(g_,state_size = None):
    """
    Returns a graph full of zeros with all (vector) graph attributes being the same size as g_,
    or all being of size "state_size".
    
    operates on tensor dictionaries which correspond to graph tuples.
    expects all attributes to be present.
    
    use `tf_gnns.td_ops._zero_graph_tf` for traceable (graph mode) functions which uses tf.shape().
    """
    g_copy = g_.copy()
    if state_size is None:
        g_copy['nodes'] = tf.zeros_like(g_['nodes'])
        g_copy['edges'] = tf.zeros_like(g_['edges'])
        g_copy['global_attr'] = tf.zeros_like(g_['global_attr'])
    else:
        g_copy['nodes'] = tf.zeros([g_['nodes'].shape[0], state_size], tf.float32)
        g_copy['edges'] = tf.zeros([g_['edges'].shape[0], state_size], tf.float32)
        g_copy['global_attr'] = tf.zeros([g_['global_attr'].shape[0], state_size], dtype = tf.float32)
    
    return g_copy

def _zero_graph_tf(g_,state_size = None):
    g_copy = g_.copy()
    if state_size is None:
        g_copy['nodes'] = tf.zeros_like(g_['nodes'])
        g_copy['edges'] = tf.zeros_like(g_['edges'])
        g_copy['global_attr'] = tf.zeros_like(g_['global_attr'])
    else:
        g_copy['nodes'] = tf.zeros([tf.shape(g_['nodes'])[0], state_size], tf.float32)
        g_copy['edges'] = tf.zeros([tf.shape(g_['edges'])[0], state_size], tf.float32)
        g_copy['global_attr'] = tf.zeros([tf.shape(g_['global_attr'])[0], state_size], dtype = tf.float32)
    
    return g_copy

def _add_gt(g1,g2):
    """
    Copies the graph structure of the first graph 
    and adds all the graph attributes.
    """
    s = _copy_structure(g1)
    nodes_ = g1['nodes'] + g2['nodes']
    edges_ = g1['edges'] + g2['edges']
    global_attr_ = g1['global_attr'] + g2['global_attr']
    s.update({'nodes' : nodes_, 'edges' : edges_ , 'global_attr' : global_attr_})
    return s


def _concat_tensordicts(t1,t2):
    td_new = _copy_structure(t1)
    td_new['nodes'] = tf.concat([t1['nodes'], t2['nodes']], axis = -1)
    td_new['edges'] = tf.concat([t1['edges'], t2['edges']], axis = -1)
    if 'global_attr' in t1:
        td_new['global_attr'] = tf.concat([t1['global_attr'], t2['global_attr']], axis = -1)
    return td_new

def _copy_structure(g_):
    """
    Returns a new dictionary containg copies of the 
    book-keeping fields of g_.
    """
    td_new = {k : g_[k] for k in GRAPH_TUPLE_STRUCTURE}
    return td_new


def _assign_add_tensor_dict(d_,od):
    """
    add nodes/edges/globals of d2 to d1 and return.
    """
    d_['nodes']       = d_['nodes']       + od['nodes']
    d_['edges']       = d_['edges']       + od['edges']
    d_['global_attr'] = d_['global_attr'] + od['global_attr']
    return d_


def _slice_conc_tensordict(td_, node_slices , edge_slices , glob_slices ):
    """
    Slices a tensor dictionary to a set of dictionaries,
    according to node_slices, edge_slices, glob_slices,
    to a set of tensor dictionaries that contain sliced ndoes, edges, globals
    and the same structure as the original tensor dictionary.
    Example:
     > # input: td with edges, nodes and glob size 32:
     > sliced_td = _slice_conc_tensordict(td, tf.constant([11,21]),tf.constant([10,22]),tf.constant([10,22]))
    """
    tds_ = [];
    assert(len(node_slices) == len(edge_slices))
    assert(len(edge_slices) == len(glob_slices))
    cni,cei, cgi = [0,0,0]
    num_slices = len(node_slices)
    for k in range(num_slices):
        ni, ei, gi = [node_slices[k], edge_slices[k], glob_slices[k]]
        new_tds = _copy_structure(td_)
        new_tds['nodes'] = td_['nodes'][:,cni:cni+ni]
        new_tds['edges'] = td_['edges'][:,cei:cei+ei]
        new_tds['global_attr'] = td_['global_attr'][:,cgi:cgi+gi]
        tds_.append(new_tds)
        cei += ei
        cni += ni
        cgi += gi
        
    return tds_

