"""Dataset loading and conversion utilities for ogbg-molhiv MPNN benchmarks."""

from __future__ import annotations

from benchmarks.ogbg_molhiv.data import DatasetSplits, load_ogbg_molhiv

import numpy as np


def validate_dataset_indexed_graphs(dataset, indices: np.ndarray, max_graphs: int = 64) -> None:
    if len(indices) == 0:
        raise ValueError("No indices provided for validation")
    check_n = min(len(indices), max_graphs)
    for i in range(check_n):
        g, y = dataset[int(indices[i])]
        if "node_feat" not in g or "edge_index" not in g:
            raise ValueError("OGB graph missing expected keys")
        if "edge_feat" not in g:
            raise ValueError("OGB graph missing edge_feat required for MPNN")
        if g["node_feat"].ndim != 2:
            raise ValueError("node_feat expected rank-2")
        if g["edge_feat"].ndim != 2:
            raise ValueError("edge_feat expected rank-2")
        if g["edge_index"].shape[0] != 2:
            raise ValueError("edge_index expected shape [2, E]")
        if int(g["edge_feat"].shape[0]) != int(g["edge_index"].shape[1]):
            raise ValueError("edge_feat length must match number of edges")
        if y is None:
            raise ValueError("label is missing")


def make_framework_samples(
    dataset,
    indices: np.ndarray,
    max_graphs: int = 32,
    include_tf: bool = True,
    include_dgl: bool = True,
):
    """Create aligned per-framework sample objects with edge features."""
    import torch
    from torch_geometric.data import Data
    dgl = None
    if include_dgl:
        import dgl  # type: ignore[no-redef]
    tf = None
    if include_tf:
        import tensorflow as tf  # type: ignore[no-redef]

    selected = indices[:max_graphs]
    tf_samples = []
    pyg_samples = []
    dgl_samples = []

    for idx in selected:
        graph, y = dataset[int(idx)]
        edge_index = graph["edge_index"]
        node_feat = graph["node_feat"].astype(np.float32)
        edge_feat = graph["edge_feat"].astype(np.float32)
        y_val = np.asarray(y, dtype=np.float32).reshape(-1)

        senders = edge_index[0].astype(np.int32)
        receivers = edge_index[1].astype(np.int32)
        n_nodes = int(graph["num_nodes"])
        n_edges = int(edge_index.shape[1])

        if include_tf:
            td = {
                "nodes": tf.convert_to_tensor(node_feat),
                "edges": tf.convert_to_tensor(edge_feat),
                "senders": tf.convert_to_tensor(senders, dtype=tf.int32),
                "receivers": tf.convert_to_tensor(receivers, dtype=tf.int32),
                "n_edges": tf.convert_to_tensor([n_edges], dtype=tf.int32),
                "n_nodes": tf.convert_to_tensor([n_nodes], dtype=tf.int32),
                "n_graphs": tf.convert_to_tensor(1, dtype=tf.int32),
                "global_attr": tf.zeros((1, 1), dtype=tf.float32),
                "global_reps_for_edges": tf.zeros((n_edges,), dtype=tf.int32),
                "global_reps_for_nodes": tf.zeros((n_nodes,), dtype=tf.int32),
            }
            tf_samples.append((td, tf.convert_to_tensor(y_val, dtype=tf.float32)))

        pyg_samples.append(
            Data(
                x=torch.tensor(node_feat, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_feat, dtype=torch.float32),
                y=torch.tensor(y_val, dtype=torch.float32),
            )
        )

        if include_dgl:
            dg = dgl.graph((senders, receivers), num_nodes=n_nodes)
            dg.ndata["x"] = torch.tensor(node_feat, dtype=torch.float32)
            dg.edata["edge_attr"] = torch.tensor(edge_feat, dtype=torch.float32)
            dg.ndata["batch"] = torch.zeros((n_nodes,), dtype=torch.long)
            dg.y = torch.tensor(y_val, dtype=torch.float32)
            dgl_samples.append(dg)

    return tf_samples, pyg_samples, dgl_samples


def validate_framework_sample_alignment(tf_samples, pyg_samples, dgl_samples) -> None:
    if tf_samples:
        if dgl_samples:
            if not (len(tf_samples) == len(pyg_samples) == len(dgl_samples)):
                raise ValueError("Sample count mismatch across frameworks")
        elif len(tf_samples) != len(pyg_samples):
            raise ValueError("Sample count mismatch between tf_gnns and PyG")
    elif dgl_samples and len(pyg_samples) != len(dgl_samples):
        raise ValueError("Sample count mismatch between PyG and DGL")

    if not tf_samples and dgl_samples:
        for pyg, dg in zip(pyg_samples, dgl_samples):
            if int(pyg.x.shape[0]) != int(dg.num_nodes()):
                raise ValueError("Node-count mismatch between PyG and DGL")
            if int(pyg.edge_index.shape[1]) != int(dg.num_edges()):
                raise ValueError("Edge-count mismatch between PyG and DGL")
        return
    if not tf_samples:
        return

    for idx, ((td, _), pyg) in enumerate(zip(tf_samples, pyg_samples)):
        tf_nodes = int(td["nodes"].shape[0])
        tf_edges = int(td["senders"].shape[0])
        if tf_nodes != int(pyg.x.shape[0]):
            raise ValueError("Node-count mismatch between tf_gnns and PyG")
        if tf_edges != int(pyg.edge_index.shape[1]):
            raise ValueError("Edge-count mismatch between tf_gnns and PyG")
        if dgl_samples:
            dg = dgl_samples[idx]
            if tf_nodes != int(dg.num_nodes()):
                raise ValueError("Node-count mismatch across framework samples")
            if tf_edges != int(dg.num_edges()):
                raise ValueError("Edge-count mismatch across framework samples")
        if int(td["edges"].shape[0]) != int(pyg.edge_attr.shape[0]):
            raise ValueError("Edge-feature count mismatch between tf_gnns and PyG")


__all__ = [
    "DatasetSplits",
    "load_ogbg_molhiv",
    "make_framework_samples",
    "validate_dataset_indexed_graphs",
    "validate_framework_sample_alignment",
]
