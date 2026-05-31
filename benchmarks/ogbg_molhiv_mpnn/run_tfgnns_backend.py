"""Run tf_gnns MPNN benchmark for one Keras backend/mode pair."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence

import numpy as np

try:
    from tqdm import trange
except Exception:
    trange = None


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


def _make_samples_raw(dataset, indices, max_graphs):
    selected = indices[:max_graphs]
    samples = []
    for idx in selected:
        graph, y = dataset[int(idx)]
        edge_index = graph["edge_index"]
        node_feat = graph["node_feat"].astype(np.float32)
        edge_feat = graph["edge_feat"].astype(np.float32)

        senders = edge_index[0]
        receivers = edge_index[1]
        n_nodes = int(graph["num_nodes"])
        n_edges = int(edge_index.shape[1])

        label = np.asarray(y, dtype=np.float32).reshape(1, 1)
        samples.append((node_feat, edge_feat, senders, receivers, n_nodes, n_edges, label))
    return samples


def _materialize_samples(raw_samples, feature_dtype, index_dtype):
    import keras

    samples = []
    for node_feat, edge_feat, senders, receivers, n_nodes, n_edges, label in raw_samples:
        td = {
            "nodes": keras.ops.convert_to_tensor(node_feat, dtype=feature_dtype),
            "edges": keras.ops.convert_to_tensor(edge_feat, dtype=feature_dtype),
            "senders": keras.ops.convert_to_tensor(senders, dtype=index_dtype),
            "receivers": keras.ops.convert_to_tensor(receivers, dtype=index_dtype),
            "n_edges": keras.ops.convert_to_tensor([n_edges], dtype=index_dtype),
            "n_nodes": keras.ops.convert_to_tensor([n_nodes], dtype=index_dtype),
            "n_graphs": keras.ops.convert_to_tensor(1, dtype=index_dtype),
            "global_attr": keras.ops.zeros((1, 1), dtype=feature_dtype),
            "global_reps_for_edges": keras.ops.zeros((n_edges,), dtype=index_dtype),
            "global_reps_for_nodes": keras.ops.zeros((n_nodes,), dtype=index_dtype),
        }
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
    try:
        jax.block_until_ready(jax.numpy.asarray(model.head.kernel))
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
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from ogb.graphproppred import GraphPropPredDataset

    dataset = GraphPropPredDataset(name="ogbg-molhiv", root=args.dataset_root)
    split = dataset.get_idx_split()
    train_idx = np.asarray(split["train"])
    raw_samples = _make_samples_raw(dataset, train_idx, max_graphs=args.max_graphs)

    import keras

    from tf_gnns.models.graphnet import GraphNetMPNN_MLP

    np.random.seed(args.seed)

    effective_index_dtype = "int64" if (args.index_dtype == "auto" and args.backend == "torch") else args.index_dtype
    if effective_index_dtype == "auto":
        effective_index_dtype = "int32"

    samples = _materialize_samples(raw_samples, args.feature_dtype, effective_index_dtype)

    class TfGNNsMPNN(keras.Model):
        def __init__(self, hidden_dim):
            super().__init__()
            self.mpnn = GraphNetMPNN_MLP(units=hidden_dim, core_units=hidden_dim, core_steps=2)
            self.head = keras.layers.Dense(1, dtype=args.feature_dtype)

        def build(self, input_shape):
            self.mpnn.build(input_shape)
            node_out_dim = self.mpnn.node_output_size
            self.head.build((None, node_out_dim))
            super().build(input_shape)

        def call(self, td, training=False):
            del training
            td = self.mpnn(td)
            pooled = keras.ops.mean(td["nodes"], axis=0, keepdims=True)
            return self.head(pooled)

    model = TfGNNsMPNN(args.hidden_dim)

    print(
        f"[tf_gnns] starting backend={args.backend} mode={args.mode} "
        f"warmup={args.warmup} steps={args.steps}"
    )

    def pick(i):
        if args.sample_mode == "single":
            return samples[0]
        return samples[i % len(samples)]

    if args.backend == "torch":
        import torch

        td0, _ = pick(0)
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

    elif args.backend == "jax":
        import jax

        td0, _ = pick(0)
        input_shape = {
            k: (None if v is None else tuple(v.shape))
            for k, v in td0.items()
        }
        model.build(input_shape)
        _ = model(td0)

        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        optimizer.build(model.trainable_variables)

        trainable_vars = [v.value for v in model.trainable_variables]
        non_trainable_vars = [v.value for v in model.non_trainable_variables]
        optimizer_vars = [v.value for v in optimizer.variables]
        loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

        def _jax_step(trainable_vars, non_trainable_vars, optimizer_vars, td, y):
            def _loss_and_state(tv, ntv):
                logits, new_ntv = model.stateless_call(tv, ntv, td, training=True)
                loss = loss_fn(y, logits)
                return loss, new_ntv

            (loss, new_ntv), grads = jax.value_and_grad(_loss_and_state, has_aux=True)(
                trainable_vars, non_trainable_vars
            )
            new_ov, new_tv = optimizer.stateless_apply(optimizer_vars, grads, trainable_vars)
            return loss, new_tv, new_ntv, new_ov

        jax_step = jax.jit(_jax_step) if args.mode == "compiled" else _jax_step

        def train_step(td, y):
            nonlocal trainable_vars, non_trainable_vars, optimizer_vars
            loss, trainable_vars, non_trainable_vars, optimizer_vars = jax_step(
                trainable_vars, non_trainable_vars, optimizer_vars, td, y
            )
            return loss

    else:
        td0, _ = pick(0)
        input_shape = {
            k: (None if v is None else tuple(v.shape))
            for k, v in td0.items()
        }
        model.build(input_shape)
        _ = model(td0)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            run_eagerly=(args.mode == "eager"),
            jit_compile=(args.mode == "compiled"),
        )

        def train_step(td, y):
            return model.train_on_batch(td, y)

    warmup_iter = range(args.warmup)
    if trange is not None and args.warmup > 0:
        warmup_iter = trange(args.warmup, desc="warmup", leave=False)
    for i in warmup_iter:
        td, y = pick(i)
        train_step(td, y)

    times = []
    step_iter = range(args.steps)
    if trange is not None and args.steps > 0:
        step_iter = trange(args.steps, desc="bench", leave=False)
    for i in step_iter:
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
        "param_dtype": str(model.head.kernel.dtype),
        "requested_index_dtype": args.index_dtype,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
