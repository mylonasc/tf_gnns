import unittest 

# TODO: Write a test on gradient computation.

class TestGraphDatastructures(unittest.TestCase):

    def test_construct_nodes_edges_simple_graph_np(self):
        """
        Tests the construction of some basic datastructures useful for GraphNet computation
        """
        n1 = Node(np.random.randn(10,10))
        n2 = Node(np.random.randn(10,10))
        e12 = Edge(np.random.randn(5,10),n1,n2)
        g = Graph([n1,n2], [e12])

        
    def test_node_operations(self):

        r1 = np.random.randn(10,10)
        r2 = np.random.randn(10,10)
        n1 = Node(r1)
        n2 = Node(r2)
        n3 = n1  + n2
        self.assertEqual(np.linalg.norm(n2.node_attr_tensor + n1.node_attr_tensor-n3.node_attr_tensor),0)

    def test_node_copy(self):
        """
        test that when copying the object the value is coppied but not the 
        reference
        """
        n1 = Node(np.random.randn(10,10))
        n2 = n1.copy()
        self.assertTrue(n1 != n2)
        self.assertTrue(np.linalg.norm((n1 - n2).node_attr_tensor)== 0.)

    def test_graph_tuple_construction(self):
        """
        Tests if I can properly set and then retrieve a graph tuple.
        """
        batch_size = 1
        node_input_size = 2
        edge_input_size = 2
        n1 = Node(np.random.randn(batch_size,node_input_size))
        n2 = Node(np.random.randn(batch_size, node_input_size))
        n3 = Node(np.random.randn(batch_size, node_input_size))
        n4 = Node(np.random.randn(batch_size, node_input_size))
        n5 = Node(np.random.randn(batch_size, node_input_size))

        e12 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n1,node_to = n2)
        e21 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n2,node_to = n1)
        e23 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n2,node_to = n3)
        e34 = Edge(np.random.randn(batch_size, edge_input_size), node_from = n3, node_to = n4)
        e45 = Edge(np.random.randn(batch_size, edge_input_size), node_from = n4, node_to = n5)

        g1 = Graph([n1,n2],[e12])
        g2 = Graph([n1,n2,n3,n4],[e12,e21,e23,e34])
        g3 = Graph([n3, n4] , [e34])

        from tf_gnns import GraphTuple, make_graph_tuple_from_graph_list # the current folder is the module.
        old_graphs_list = [g1.copy(),g2.copy(),g3.copy()]
        graph_tuple = make_graph_tuple_from_graph_list(old_graphs_list)
        new_graphs_list = [graph_tuple.get_graph(k) for k in range(graph_tuple.n_graphs)]
        self.assertTrue(np.all([(k.is_equal_by_value(m) and k.compare_connectivity(m) ) for k, m in zip(new_graphs_list, old_graphs_list)]))

class TestGraphNet(unittest.TestCase):
    def test_construct_simple_eval_graphnet(self):
        from tf_gnns import GraphNet, make_keras_simple_agg
        edge_input_size = 15
        node_input_size = 11
        node_output_size, edge_output_size = node_input_size, edge_input_size
        node_input = tf.keras.layers.Input(shape = (node_input_size,))
        edge_input = tf.keras.layers.Input(shape = (edge_input_size,))

        node_function = tf.keras.Model(outputs = tf.keras.layers.Dense(node_output_size)(node_input), inputs= node_input)
        edge_function = tf.keras.Model(outputs = tf.keras.layers.Dense(edge_output_size)(edge_input), inputs= edge_input)
        edge_aggregation_function = make_keras_simple_agg(edge_output_size,'mean')
        graphnet = GraphNet(node_function = node_function, edge_function = edge_function, edge_aggregation_function = edge_aggregation_function)
        batch_size = 10
        n1 = Node(np.random.randn(batch_size,node_input_size))
        n2 = Node(np.random.randn(batch_size, node_input_size))
        e12 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n1,node_to = n2)
        g = Graph([n1,n2],[e12])

    def test_mean_max_aggregator(self):
        """
        Tests if the special compound aggregator which outputs a concatenation of mean and max works.
        """
        from tf_gnns import GraphNet, _aggregation_function_factory
        node_input_size = 5
        edge_input_size = 5
        edge_output_size = 5
        message_shape = 2*edge_output_size #<- message size is larger because it is a concatenation of two aggregators.
        batch_size = 6

        # The naive implementation:
        v1,v2,v3 = [np.ones([batch_size, node_input_size])*k for k in range(3)]
        n1 , n2, n3 = [Node(v_) for v_ in [v1,v2,v3]]
        e21 = Edge(np.ones([batch_size, node_input_size])*0., node_from = n2, node_to = n1)
        e31 = Edge(np.ones([batch_size, node_input_size])*10, node_from = n3, node_to = n1)

        #** The "None" is the actual batch dimension during computation with the Naive evaluators 
        # ("safe" and "batched"). Reduction happens w.r.t. 1st dimension which enumerates incoming edges.
        edge_aggregation_function = _aggregation_function_factory((None, edge_output_size), agg_type = 'mean_max')

        node_input = tf.keras.layers.Input(shape = (node_input_size,), name = "node_state")
        agg_edge_state_input = tf.keras.layers.Input(shape = (message_shape,), name = "edge_state_agg")
        edge_input = tf.keras.layers.Input(shape = (edge_input_size,), name = "edge_state")

        node_function = tf.keras.Model(outputs = tf.identity(node_input),
                inputs = [agg_edge_state_input, node_input],name = "node_function")

        edge_function = tf.keras.Model(outputs = tf.identity(edge_input),
                inputs = [edge_input])

        gn = GraphNet(node_function = node_function, 
                edge_function = edge_function, 
                edge_aggregation_function = edge_aggregation_function)
        g = Graph([n1,n2,n3],[e21, e31])
        g_, m = gn.graph_eval(g, eval_mode=  "safe", return_messages = True)
        m_correct = np.hstack([np.ones([batch_size,edge_output_size])*5, np.ones([batch_size,edge_output_size])*10.])
        self.assertTrue(np.all(m == m_correct))


    def test_eval_modes(self):
        """
        test the different evaluation modes.
        There are 3 evaluation modes - one appropriate for batched graphs, and two for graphs of the same shape ("batched" or unbached ("safe")).
        The "safe" mode is used as reference for the correct results; All modes should give the same output within an error margin (due to finite precission 
        rounding errors and the different comp. graphs.)
        """
        from tf_gnns import GraphNet, make_mlp_graphnet_functions
        import code

        batch_size = 12
        tf.keras.backend.set_floatx("float64")
        node_input_size = 10
        edge_input_size = node_input_size

        n1 = Node(np.random.randn(batch_size,node_input_size))
        n2 = Node(np.random.randn(batch_size, node_input_size))
        n3 = Node(np.random.randn(batch_size, node_input_size))
        node_abs_vals = [np.abs(n.node_attr_tensor) for n in [n1,n2,n3]]

        e12 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n1,node_to = n2)
        e21 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n2,node_to = n1)
        e23 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n2,node_to = n3)
        edge_abs_vals = [np.abs(e.edge_tensor) for e in [e12,e21,e23]]

        g1 = Graph([n1,n2,n3],[e12,e21,e23])


        node_output_size = 17
        ## The non-graph independent version:
        gi = False
        graph_fcn = make_mlp_graphnet_functions(150, 
                                            node_input_size = node_input_size, 
                                            node_output_size = node_output_size, 
                                            graph_indep=False)

        gn = GraphNet(**graph_fcn )
        res1 = gn.graph_eval(g1.copy(),eval_mode = "safe")
        res2 = gn.graph_eval(g1.copy(), eval_mode = "batched")

        error_nodes = np.max([np.linalg.norm(n1_.node_attr_tensor - n2_.node_attr_tensor) for n1_, n2_ in zip(res1.nodes, res2.nodes)])/np.min(node_abs_vals)
        error_edges = np.max([np.linalg.norm(e1_.edge_tensor - e2_.edge_tensor) for e1_,e2_ in zip(res1.edges, res2.edges)])/np.min(edge_abs_vals)
        #print(error_nodes, error_edges)
        self.assertTrue(error_nodes < 1e-10)
        self.assertTrue(error_edges < 1e-10)

        ## The graph-independent version:
        gi = True
        graph_fcn = make_mlp_graphnet_functions(150, 
                node_input_size = node_input_size,
                node_output_size = node_input_size, 
                graph_indep=gi, use_edge_state_agg_input = False)
        graph_fcn.update({"graph_independent" : gi})
        gn = GraphNet(**graph_fcn )
        res1 = gn.graph_eval(g1.copy(),eval_mode = "safe")
        res2 = gn.graph_eval(g1.copy(), eval_mode = "batched")
        error_nodes = np.max([np.linalg.norm(n1.node_attr_tensor - n2.node_attr_tensor) for n1, n2 in zip(res1.nodes, res2.nodes)])/np.min(node_abs_vals)
        error_edges = np.max([np.linalg.norm(e1.edge_tensor - e2.edge_tensor) for e1,e2 in zip(res1.edges, res2.edges)])/np.min(edge_abs_vals)
        self.assertTrue(error_nodes < 1e-10)
        self.assertTrue(error_edges < 1e-10)

    def test_save_load(self):
        # TODO: this needs to be updated for the global blocks (there are 3 more functions to be treated). 
        #       Write another test and keep this one, in order to keep backwards compatibility.
        from tf_gnns import make_mlp_graphnet_functions, GraphNet
        graph_fcn = make_mlp_graphnet_functions(150, node_input_size = 10, node_output_size = 10, graph_indep=False)
        gn = GraphNet(**graph_fcn)
        gn.save("/tmp/test_gn")
        gn_loaded = GraphNet.make_from_path("/tmp/test_gn")
        self.assertTrue(np.all([np.sum(np.abs(w1 - w2))<1e-10 for w1,w2 in zip(gn.weights,gn_loaded.weights)]))

    def test_graph_tuple_eval(self):
        """
        The graph tuples are graphs of different sizes batched to a single object,
        to allow for more single-instruction multiple-data computation (batched computation).
        This is the only evalution mode DeepMind's graphnets implement directly. 
        This mode is much more computationally efficient.
        This mode allows computation with unsorted segment sum aggregators.
        """
        import code


        ## Constructing a graph tuple:

        batch_size = 1
        node_input_size = 10
        edge_input_size = 10
        n1 = Node(np.random.randn(batch_size,node_input_size))
        n2 = Node(np.random.randn(batch_size, node_input_size))
        n3 = Node(np.random.randn(batch_size, node_input_size))
        n4 = Node(np.random.randn(batch_size, node_input_size))
        n5 = Node(np.random.randn(batch_size, node_input_size))

        e12 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n1,node_to = n2)
        e21 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n2,node_to = n1)
        e23 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n2,node_to = n3)
        e34 = Edge(np.random.randn(batch_size, edge_input_size), node_from = n3, node_to = n4)
        e45 = Edge(np.random.randn(batch_size, edge_input_size), node_from = n4, node_to = n5)

        g1 = Graph([n1,n2],[e12]).copy()
        g2 = Graph([n1,n2,n3,n4],[e12,e21,e23,e34]).copy()
        g3 = Graph([n3, n4] , [e34]).copy()

        from tf_gnns import GraphTuple, make_graph_tuple_from_graph_list ,GraphNet , make_mlp_graphnet_functions# the current folder is the module.
        old_graphs_list = [g1.copy(),g2.copy(),g3.copy()]
        graph_tuple = make_graph_tuple_from_graph_list(old_graphs_list)
        #new_graphs_list = [graph_tuple.get_graph(k) for k in range(graph_tuple.n_graphs)]
        #self.assertTrue(np.all([(k.is_equal_by_value(m) and k.compare_connectivity(m) ) for k, m in zip(new_graphs_list, old_graphs_list)]))

        graph_fcn = make_mlp_graphnet_functions(150, node_input_size = 10, node_output_size = 10, graph_indep=False, aggregation_function = "mean")
        gn = GraphNet(**graph_fcn)
        gt_copy = graph_tuple.copy()
        gn.graph_tuple_eval(gt_copy)
        graphs_evaluated_separately = [gn.graph_eval(g_)  for g_ in old_graphs_list]
        graphs_evaluated_from_graph_tuple = [gt_copy.get_graph(i) for i in range(gt_copy.n_graphs)]
        flatten_nodes = lambda x : tf.stack([x_.get_state() for x_ in x.nodes])
        flatten_edges = lambda x : tf.stack([x_.edge_tensor for x_ in x.edges])
        for g1,g2 in zip(graphs_evaluated_from_graph_tuple, graphs_evaluated_separately):
            self.assertTrue(tf.norm(flatten_nodes(g1)- flatten_nodes(g2))<1e-10)
            self.assertTrue(tf.norm(flatten_edges(g1) - flatten_edges(g2)) < 1e-10)

    def test_graph_tuple_eval_with_global(self):
        """
        Test if the evaluation of graph tuples with global variables works.
        """

        ## Constructing a graph tuple:

        batch_size = 1
        node_input_size = 10
        edge_input_size = 10
        global_attr_size  = 5
        n1 = Node(np.random.randn(batch_size,node_input_size))
        n2 = Node(np.random.randn(batch_size, node_input_size))
        n3 = Node(np.random.randn(batch_size, node_input_size))
        n4 = Node(np.random.randn(batch_size, node_input_size))
        n5 = Node(np.random.randn(batch_size, node_input_size))

        e12 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n1,node_to = n2)
        e21 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n2,node_to = n1)
        e23 = Edge(np.random.randn(batch_size, edge_input_size),node_from = n2,node_to = n3)
        e34 = Edge(np.random.randn(batch_size, edge_input_size), node_from = n3, node_to = n4)
        e45 = Edge(np.random.randn(batch_size, edge_input_size), node_from = n4, node_to = n5)

        g1 = Graph([n1,n2],[e12]).copy()
        g2 = Graph([n1,n2,n3,n4],[e12,e21,e23,e34]).copy()
        g3 = Graph([n3, n4] , [e34]).copy()

        from tf_gnns import GraphTuple, make_graph_tuple_from_graph_list ,GraphNet , make_mlp_graphnet_functions# the current folder is the module.
        old_graphs_list = [g1.copy(),g2.copy(),g3.copy()]
        graph_tuple = make_graph_tuple_from_graph_list(old_graphs_list)
        global_vars = tf.Variable(np.random.randn(graph_tuple.n_graphs,global_attr_size))
        global_out = 10

        graph_fcn = make_mlp_graphnet_functions(150,
                node_input_size = 10,
                node_output_size = 10,
                graph_indep=False,
                use_global_to_edge = True,
                use_global_to_node = True, 
                use_global_input = True,
                global_input_size = global_attr_size, 
                global_output_size = 10,
                create_global_function = True)

        gn = GraphNet(**graph_fcn)
        gt_copy = graph_tuple.copy()

        ## This is how a global is assigned. The "update.." creates some flat vectors useful 
        #  for the segment sums and reshaping of the tensors (when/in they are used in the 
        #  node and edge computations)
        gt_copy.assign_global(global_vars)
        gt_copy.update_reps_for_globals()
        out = gn.graph_tuple_eval(gt_copy )#, global_vars)


    def test_computation_graph_to_global(self):
        """
        Tests the construction of a simple GraphTuple without a global attribute
        and its computation with a full GN (without a global input)
        """
        import tensorflow as tf
        import numpy as np

        from tf_gnns import GraphTuple
        from tf_gnns.graphnet_utils import _aggregation_function_factory

        from tf_gnns import make_mlp_graphnet_functions, GraphNet, Node, Edge, Graph, GraphTuple, make_graph_tuple_from_graph_list

        ## Testing GN that contains globals:

        ### Create an encode-core-decode network:

        ## Create a GraphTuple to compute with:
        node_state_size=4;
        edge_state_size = 10;
        ngraphs = 16;

        graphs = [];
        for gr in range(ngraphs):
            n1 = Node(np.random.randn(1, node_state_size))
            n2 = Node(np.random.randn(1, node_state_size))
            n3 = Node(np.random.randn(1, node_state_size))
            e12 = Edge(np.random.randn(1, edge_state_size) , node_from=n1,node_to=n2)
            e13 = Edge(np.random.randn(1, edge_state_size) , node_from=n1,node_to=n2)
            e23 = Edge(np.random.randn(1, edge_state_size) , node_from=n1,node_to=n2)
            graphs.append(Graph([n1,n2,n3],[e12,e13,e23]))

        gt = make_graph_tuple_from_graph_list(graphs    )

        units = 45
        gi_node_input_size = node_state_size
        gi_edge_input_size = edge_state_size
        gn_core_size = 15
       
        ## Creation of a graph-to-global network (without global in the input side:)
        gn_input = make_mlp_graphnet_functions(45,
                                    gi_node_input_size,
                                    gn_core_size,
                                    edge_input_size=gi_edge_input_size,
                                    edge_output_size=gn_core_size,
                                    create_global_function = True,
                                    global_input_size=None,
                                    use_global_input= False,
                                    global_output_size = gn_core_size,
                                    graph_indep = False)

        
        gn_constr_input_named_params = ['edge_function', 
                'global_function', 
                'node_function',
                'edge_aggregation_function', 
                'node_to_global_aggregation_function',
                'graph_independent','use_global_input']

        correct_keys = np.all([k in gn_constr_input_named_params for k in gn_input.keys()])
        self.assertTrue(correct_keys)
        gn_gi = GraphNet(**gn_input)
        gt_out = gn_gi.graph_tuple_eval(gt.copy())
        self.assertTrue(gt_out.global_attr.shape == (ngraphs,gn_core_size))

class TestTraced_eval(unittest.TestCase):
    def test_correct_results_traced(self):
        import tensorflow as tf
        import numpy as np

        from tf_gnns import GraphTuple
        from tf_gnns.graphnet_utils import _aggregation_function_factory, make_full_graphnet_functions

        from tf_gnns import make_mlp_graphnet_functions, GraphNet, Node, Edge, Graph, GraphTuple, make_graph_tuple_from_graph_list
        import code


        ## Create a GraphTuple to compute with:
        node_state_size=4;
        edge_state_size = 10;
        ngraphs = 16;

        graphs = [];
        for gr in range(ngraphs):
            n1 = Node(np.random.randn(1, node_state_size))
            n2 = Node(np.random.randn(1, node_state_size))
            n3 = Node(np.random.randn(1, node_state_size))
            e12 = Edge(np.random.randn(1, edge_state_size) , node_from=n1,node_to=n2)
            e13 = Edge(np.random.randn(1, edge_state_size) , node_from=n1,node_to=n3)
            e23 = Edge(np.random.randn(1, edge_state_size) , node_from=n2,node_to=n3)
            graphs.append(Graph([n1,n2,n3],[e12,e13,e23]))

        gt = make_graph_tuple_from_graph_list(graphs)

        units = 45
        gi_node_input_size = node_state_size
        gi_edge_input_size = edge_state_size
        gn_core_size = 15
       
        ## Creation of a graph-to-global network (without global in the input side:)
        gn_input_args = make_mlp_graphnet_functions(45,
                                    gi_node_input_size,
                                    gn_core_size,
                                    edge_input_size=gi_edge_input_size,
                                    edge_output_size=gn_core_size,
                                    create_global_function = True,
                                    global_input_size=None,
                                    use_global_input= False,
                                    global_output_size = gn_core_size,
                                    graph_indep = False)

        gn_core_args = make_full_graphnet_functions(units, gn_core_size)

        
        gn_constr_input_named_params = ['edge_function', 
                'global_function', 
                'node_function',
                'edge_aggregation_function', 
                'node_to_global_aggregation_function',
                'graph_independent','use_global_input']

        correct_keys = np.all([k in gn_constr_input_named_params for k in gn_input_args.keys()])
        self.assertTrue(correct_keys)

        gn_gi = GraphNet(**gn_input_args)
        gn_core = GraphNet(**gn_core_args)
        gt_out_1 = gn_gi.graph_tuple_eval(gt.copy())
        gt_out = gn_core.graph_tuple_eval(gt_out_1)


        tensor_dict_out = gn_gi.eval_tensor_dict(gt.copy().to_tensor_dict())
        tensor_dict_out = gn_core.eval_tensor_dict(tensor_dict_out)
        edges_err = np.linalg.norm(gt_out.edges - tensor_dict_out['edges']) 
        nodes_err = np.linalg.norm(gt_out.nodes - tensor_dict_out['nodes']) 
        glob_err  = np.linalg.norm(gt_out.global_attr  - tensor_dict_out['global_attr'])

        self.assertTrue(edges_err < 1e-10)
        self.assertTrue(nodes_err < 1e-10)
        self.assertTrue(glob_err < 1e-10)

class TestHighLevel(unittest.TestCase):
    """
    Test for high-level classes that create composite GNs
    """
    def test_graphnet_mlp(self):
        """
        Tests the basic encode-core-decode GN:
        """
        import tensorflow as tf
        import numpy as np
        from tf_gnns import GraphNetMLP, GraphNetMPNN_MLP
        from tf_gnns import GraphTuple

        def _get_tensor_dict():
            edges = tf.constant(np.random.randn(5, 123))
            nodes = tf.constant(np.random.randn(5, 123))
            senders, receivers = [tf.constant(v) for v in [[0,0,0,0,0], [0,1,2,3,4]]]
            gt_in = GraphTuple(nodes,edges,
                senders, receivers,
                n_nodes = [nodes.shape[0]],
                n_edges = [len(receivers)]
            )
            td = gt_in.to_tensor_dict()
            return td 

        td = _get_tensor_dict()
        td['global_attr'] = [None]
        gn = GraphNetMLP(32)
        res= gn(td)

    def test_graphnet_mpnn_mlp(self):

        import tensorflow as tf
        import numpy as np
        from tf_gnns import GraphNetMLP, GraphNetMPNN_MLP
        from tf_gnns import GraphTuple

        def _get_tensor_dict():
            edges = tf.constant(np.random.randn(5, 123))
            nodes = tf.constant(np.random.randn(5, 123))
            senders, receivers = [tf.constant(v) for v in [[0,0,0,0,0], [0,1,2,3,4]]]
            gt_in = GraphTuple(nodes,edges,
                senders, receivers,
                n_nodes = [nodes.shape[0]],
                n_edges = [len(receivers)]
            )
            td = gt_in.to_tensor_dict()
            return td 

        td = _get_tensor_dict()
        td['global_attr'] = [None]
        gn = GraphNetMLP(32)
        res= gn(td)

        td = _get_tensor_dict()
        gn2 = GraphNetMPNN_MLP(32)
        res= gn2(td)


if __name__ == "__main__":

    from tf_gnns import Node, Edge, Graph
    import tensorflow as tf
    import numpy as np
    unittest.main(verbosity = 2)

