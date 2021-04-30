# Set this env. variable first, since it's needed from some classes:
import os
os.environ['TFGNNS_HTML_ASSETS']=os.path.join(__file__.strip('__init__.py'), 'assets','html_css')

from .datastructures import Graph, GraphTuple, Node, Edge, make_graph_tuple_from_graph_list
from .graphnet_utils import GraphNet, make_node_mlp, make_edge_mlp, make_keras_simple_agg, make_mlp_graphnet_functions, make_global_mlp
from .graphnet_utils import _aggregation_function_factory, make_full_graphnet_functions, make_graph_indep_graphnet_functions, make_graph_to_graph_and_global_functions
print("loaded tfgnn lib")
