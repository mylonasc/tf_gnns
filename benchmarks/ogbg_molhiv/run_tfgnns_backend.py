"""Run tf_gnns benchmark for one Keras backend/mode pair.

This script runs in a separate process so each invocation can set
``KERAS_BACKEND`` before importing Keras/tf_gnns.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["tensorflow", "torch", "jax"])
    p.add_argument("--mode", required=True, choices=["eager", "compiled"])
    p.add_argument("--dataset-root", default="dataset")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--max-graphs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-mode", choices=["single", "cycle"], default="single")
    p.add_argument("--feature-dtype", default="float32")
    p.add_argument("--index-dtype", default="auto")
    return p.parse_args()


def _make_tfgnns_samples(dataset, indices, max_graphs, feature_dtype, index_dtype):
    import keras

    selected = indices[:max_graphs]
    samples = []
    for idx in selected:
        graph, y = dataset[int(idx)]
        edge_index = graph["edge_index"]
        node_feat = graph["node_feat"]

        senders = edge_index[0]
        receivers = edge_index[1]
        n_nodes = int(graph["num_nodes"])
        n_edges = int(edge_index.shape[1])

        td = {
            "nodes": keras.ops.convert_to_tensor(node_feat, dtype=feature_dtype),
            "edges": keras.ops.zeros((n_edges, 1), dtype=feature_dtype),
            "senders": keras.ops.convert_to_tensor(senders, dtype=index_dtype),
            "receivers": keras.ops.convert_to_tensor(receivers, dtype=index_dtype),
            "n_edges": keras.ops.convert_to_tensor([n_edges], dtype=index_dtype),
            "n_nodes": keras.ops.convert_to_tensor([n_nodes], dtype=index_dtype),
            "n_graphs": keras.ops.convert_to_tensor(1, dtype=index_dtype),
            "global_attr": None,
            "global_reps_for_edges": keras.ops.zeros((n_edges,), dtype=index_dtype),
            "global_reps_for_nodes": keras.ops.zeros((n_nodes,), dtype=index_dtype),
            "edge_weights": keras.ops.ones((n_edges,), dtype=feature_dtype),
        }
        label = np.asarray(y, dtype=np.float32).reshape(1, 1)
        samples.append((td, keras.ops.convert_to_tensor(label, dtype=feature_dtype)))
    return samples


def _block_jax_ready(value, model):
    import jax

    def _block(v):
        if isinstance(v, Mapping):
            for vv in v.values():
                _block(vv)
            return
        if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
            for vv in v:
                _block(vv)
            return
        try:
            jax.block_until_ready(v)
            return
        except Exception:
            pass

        try:
            vv = jax.numpy.asarray(v)
            jax.block_until_ready(vv)
        except Exception:
            pass

    _block(value)
    # Fallback: ensure updated parameter state is synchronized.
    try:
        jax.block_until_ready(jax.numpy.asarray(model.gcn1.kernel))
    except Exception:
        pass


def _detect_device(backend):
    if backend == "tensorflow":
        import tensorflow as tf

        return "gpu" if tf.config.list_physical_devices("GPU") else "cpu"
    if backend == "torch":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if backend == "jax":
        import jax

        return jax.default_backend()
    return backend


def main():
    args = parse_args()
    os.environ["KERAS_BACKEND"] = args.backend
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    import keras
    from ogb.graphproppred import GraphPropPredDataset
    from tf_gnns.models.gcn import SparseGCNConv

    np.random.seed(args.seed)

    if args.index_dtype == "auto":
        effective_index_dtype = "int64" if args.backend == "torch" else "int32"
    else:
        effective_index_dtype = args.index_dtype

    dataset = GraphPropPredDataset(name="ogbg-molhiv", root=args.dataset_root)
    split = dataset.get_idx_split()
    train_idx = np.asarray(split["train"])
    samples = _make_tfgnns_samples(
        dataset,
        train_idx,
        max_graphs=args.max_graphs,
        feature_dtype=args.feature_dtype,
        index_dtype=effective_index_dtype,
    )

    class TfGNNsGCN(keras.Model):
        def __init__(self, hidden_dim):
            super().__init__()
            self.gcn1 = SparseGCNConv(
                hidden_dim,
                activation="relu",
                feature_dtype=args.feature_dtype,
                index_dtype=effective_index_dtype,
            )
            self.gcn2 = SparseGCNConv(
                hidden_dim,
                activation="relu",
                feature_dtype=args.feature_dtype,
                index_dtype=effective_index_dtype,
            )
            self.head = keras.layers.Dense(1, dtype=args.feature_dtype)

        def call(self, td):
            td = self.gcn1(td)
            td = self.gcn2(td)
            pooled = keras.ops.mean(td["nodes"], axis=0, keepdims=True)
            return self.head(pooled)

    model = TfGNNsGCN(args.hidden_dim)

    def pick(i):
        if args.sample_mode == "single":
            return samples[0]
        return samples[i % len(samples)]

    if args.backend == "torch":
        import torch

        td0, y0 = pick(0)
        _ = model(td0)
        if args.mode == "compiled" and hasattr(torch, "compile"):
            model = torch.compile(model)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def train_step(td, y):
            opt.zero_grad(set_to_none=True)
            logits = model(td)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            return loss

    else:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            run_eagerly=(args.mode == "eager"),
            jit_compile=(args.mode == "compiled"),
        )

        def train_step(td, y):
            return model.train_on_batch(td, y)

    for i in range(args.warmup):
        td, y = pick(i)
        train_step(td, y)

    times = []
    for i in range(args.steps):
        td, y = pick(i)
        t0 = time.perf_counter() * 1e3
        out = train_step(td, y)
        if args.backend == "jax":
            _block_jax_ready(out, model)
        t1 = time.perf_counter() * 1e3
        times.append(t1 - t0)

    arr = np.array(times, dtype=np.float64)
    td0, _ = pick(0)
    _ = model(td0)
    result = {
        "framework": "tf_gnns",
        "mode": f"keras_{args.mode}",
        "device": _detect_device(args.backend),
        "backend": args.backend,
        "avg_step_ms": float(arr.mean()),
        "std_step_ms": float(arr.std()),
        "steps_per_sec": float(1000.0 / arr.mean()),
        "feature_dtype": str(td0["nodes"].dtype),
        "index_dtype": str(td0["senders"].dtype),
        "param_dtype": str(model.gcn1.kernel.dtype),
        "requested_index_dtype": args.index_dtype,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
