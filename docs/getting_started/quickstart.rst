Quickstart
==========

This example uses the high-level ``GraphNetMLP`` layer on a tensor-dictionary
graph tuple.

.. code-block:: python

   import numpy as np
   from tf_gnns.models.graphnet import GraphNetMLP

   graph = {
       "nodes": np.random.randn(4, 8).astype("float32"),
       "edges": np.random.randn(6, 8).astype("float32"),
       "senders": np.array([0, 0, 1, 2, 3, 3], dtype="int32"),
       "receivers": np.array([1, 2, 2, 3, 0, 1], dtype="int32"),
       "n_nodes": np.array([4], dtype="int32"),
       "n_edges": np.array([6], dtype="int32"),
       "global_attr": np.random.randn(1, 8).astype("float32"),
       "global_reps_for_nodes": np.zeros((4,), dtype="int32"),
       "global_reps_for_edges": np.zeros((6,), dtype="int32"),
       "n_graphs": 1,
   }

   model = GraphNetMLP(units=32, core_steps=2)
   output = model(graph)

   print(output["nodes"].shape, output["edges"].shape, output["global_attr"].shape)

For more complete workflows, see the rendered notebooks in :doc:`../tutorials/index`.
