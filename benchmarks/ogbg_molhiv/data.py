"""Dataset loading and conversion utilities for ogbg-molhiv."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np


@dataclass
class DatasetSplits:
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray


def load_ogbg_molhiv(root: str = "dataset"):
    # OGB internally calls torch.load on cached preprocessing artifacts.
    # Torch >=2.6 defaults to weights_only=True, which breaks OGB's general
    # object loading path. Force the legacy behavior for trusted local datasets.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    from ogb.graphproppred import GraphPropPredDataset

    dataset = GraphPropPredDataset(name="ogbg-molhiv", root=root)
    split = dataset.get_idx_split()
    splits = DatasetSplits(
        train_idx=np.asarray(split["train"]),
        valid_idx=np.asarray(split["valid"]),
        test_idx=np.asarray(split["test"]),
    )
    return dataset, splits


def _graph_stats_from_ogb_graph(graph: dict) -> tuple[int, int]:
    n_nodes = int(graph["num_nodes"])
    n_edges = int(graph["edge_index"].shape[1])
    return n_nodes, n_edges


def validate_dataset_indexed_graphs(dataset, indices: np.ndarray, max_graphs: int = 64) -> None:
    if len(indices) == 0:
        raise ValueError("No indices provided for validation")
    check_n = min(len(indices), max_graphs)
    for i in range(check_n):
        g, y = dataset[int(indices[i])]
        if "node_feat" not in g or "edge_index" not in g:
            raise ValueError("OGB graph missing expected keys")
        if g["node_feat"].ndim != 2:
            raise ValueError("node_feat expected rank-2")
        if g["edge_index"].shape[0] != 2:
            raise ValueError("edge_index expected shape [2, E]")
        if y is None:
            raise ValueError("label is missing")


def make_framework_samples(dataset, indices: np.ndarray, max_graphs: int = 32):
    """Create aligned per-framework sample objects for one batch-like slice."""
    import tensorflow as tf
    import torch
    import dgl
    from torch_geometric.data import Data

    selected = indices[: max_graphs]
    tf_samples = []
    pyg_samples = []
    dgl_samples = []

    for idx in selected:
        graph, y = dataset[int(idx)]
        edge_index = graph["edge_index"]
        node_feat = graph["node_feat"].astype(np.float32)
        y_val = np.asarray(y, dtype=np.float32).reshape(-1)

        senders = edge_index[0].astype(np.int32)
        receivers = edge_index[1].astype(np.int32)
        n_nodes = int(graph["num_nodes"])
        n_edges = int(edge_index.shape[1])

        td = {
            "nodes": tf.convert_to_tensor(node_feat),
            "edges": tf.zeros((n_edges, 1), dtype=tf.float32),
            "senders": tf.convert_to_tensor(senders, dtype=tf.int32),
            "receivers": tf.convert_to_tensor(receivers, dtype=tf.int32),
            "n_edges": tf.convert_to_tensor([n_edges], dtype=tf.int32),
            "n_nodes": tf.convert_to_tensor([n_nodes], dtype=tf.int32),
            "n_graphs": tf.convert_to_tensor(1, dtype=tf.int32),
            "global_attr": None,
            "global_reps_for_edges": tf.zeros((n_edges,), dtype=tf.int32),
            "global_reps_for_nodes": tf.zeros((n_nodes,), dtype=tf.int32),
            "edge_weights": tf.ones((n_edges,), dtype=tf.float32),
            "labels": tf.convert_to_tensor(y_val, dtype=tf.float32),
        }
        tf_samples.append(td)

        pyg_samples.append(
            Data(
                x=torch.tensor(node_feat, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                y=torch.tensor(y_val, dtype=torch.float32),
            )
        )

        dg = dgl.graph((senders, receivers), num_nodes=n_nodes)
        dg.ndata["x"] = torch.tensor(node_feat, dtype=torch.float32)
        dg.ndata["batch"] = torch.zeros((n_nodes,), dtype=torch.long)
        dg.y = torch.tensor(y_val, dtype=torch.float32)
        dgl_samples.append(dg)

    return tf_samples, pyg_samples, dgl_samples


def validate_framework_sample_alignment(tf_samples, pyg_samples, dgl_samples) -> None:
    if not (len(tf_samples) == len(pyg_samples) == len(dgl_samples)):
        raise ValueError("Sample count mismatch across frameworks")

    for td, pyg, dg in zip(tf_samples, pyg_samples, dgl_samples):
        tf_nodes = int(td["nodes"].shape[0])
        tf_edges = int(td["senders"].shape[0])
        if tf_nodes != int(pyg.x.shape[0]) or tf_nodes != int(dg.num_nodes()):
            raise ValueError("Node-count mismatch across framework samples")
        if tf_edges != int(pyg.edge_index.shape[1]) or tf_edges != int(dg.num_edges()):
            raise ValueError("Edge-count mismatch across framework samples")
