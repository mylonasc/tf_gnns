"""Optuna hyperparameter search for tf_gnns GCNv2 on OGBN-Arxiv.

Search space:
- learning rate: [1e-4, 5e-4, 1e-3]
- depth: [3, 5, 7]
- width: [256, 512]
- normalization: [batchnorm, layernorm, none]
- dropout: [0.1, 0.2, 0.5]

For each trial, metrics are recorded every ``eval_every`` steps:
- loss
- train_acc
- val_acc
- test_acc
- best_val
- best_test

Results are written per trial to CSV for later inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import optuna


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna search for GCNv2 on OGBN-Arxiv")
    p.add_argument("--dataset-root", type=str, default="dataset")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study-name", type=str, default="sparsegcn_ogbn_arxiv")
    p.add_argument(
        "--storage",
        type=str,
        default="sqlite:///benchmarks/results/optuna_sparsegcn_ogbn_arxiv.db",
    )
    p.add_argument("--output-dir", type=str, default="benchmarks/results/optuna_sparsegcn_ogbn_arxiv")
    p.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--fanout", type=int, default=10)
    p.add_argument("--max-subgraph-nodes", type=int, default=50000)
    p.add_argument("--early-stop-patience", type=int, default=100)
    return p.parse_args()


def prepare_data(root: str):
    from ogb.nodeproppred import NodePropPredDataset

    dataset = NodePropPredDataset(name="ogbn-arxiv", root=root)
    split_idx = dataset.get_idx_split()
    graph, labels = dataset[0]

    node_feat = graph["node_feat"].astype(np.float32)
    edge_index = graph["edge_index"]
    senders = edge_index[0].astype(np.int32)
    receivers = edge_index[1].astype(np.int32)

    n_nodes = int(node_feat.shape[0])
    n_edges = int(edge_index.shape[1])
    n_classes = int(labels.max()) + 1

    y = labels.reshape(-1).astype(np.int32)

    train_idx = split_idx["train"].astype(np.int32).reshape(-1)
    valid_idx = split_idx["valid"].astype(np.int32).reshape(-1)
    test_idx = split_idx["test"].astype(np.int32).reshape(-1)

    return {
        "node_feat": node_feat,
        "senders": senders,
        "receivers": receivers,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "y": y,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "num_classes": n_classes,
    }


def make_model(num_classes: int, depth: int, width: int, dropout: float, norm: str) -> keras.Model:
    import keras
    from tf_gnns.models.gcn import GCNv2

    if depth < 1:
        raise ValueError("depth must be >= 1")

    use_batchnorm = norm == "batchnorm"
    use_layernorm = norm == "layernorm"

    class ArxivGCN(keras.Model):
        def __init__(self):
            super().__init__()
            self.gcn = GCNv2(
                hidden_units=width,
                output_units=num_classes,
                num_layers=depth,
                input_dropout_rate=0.15,
                dropout_rate=dropout,
                add_self_loops=True,
                normalize=True,
                batchnorm=use_batchnorm,
                layernorm=use_layernorm,
            )

        def call(self, td, training=False):
            out = self.gcn(td, training=training)
            return out["nodes"]

    return ArxivGCN()


def eval_acc(evaluator: Evaluator, logits_np: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    y_true = y[idx].reshape(-1, 1)
    y_pred = np.argmax(logits_np[idx], axis=-1).reshape(-1, 1)
    return float(evaluator.eval({"y_true": y_true, "y_pred": y_pred})["acc"])


def write_trial_metrics_csv(path: Path, rows: list[dict]):
    fieldnames = ["step", "loss", "train_acc", "val_acc", "test_acc", "best_val", "best_test"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_in_neighbors(n_nodes: int, senders: np.ndarray, receivers: np.ndarray) -> list[np.ndarray]:
    buckets: list[list[int]] = [[] for _ in range(n_nodes)]
    for s, r in zip(senders.tolist(), receivers.tolist()):
        buckets[int(r)].append(int(s))
    return [np.asarray(v, dtype=np.int32) for v in buckets]


def sample_khop_subgraph(
    seed_nodes: np.ndarray,
    depth: int,
    in_neighbors: list[np.ndarray],
    senders: np.ndarray,
    receivers: np.ndarray,
    fanout: int,
    max_nodes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_order: list[int] = []
    seen = np.zeros((len(in_neighbors),), dtype=bool)

    frontier = seed_nodes.astype(np.int32, copy=False)
    for node in frontier.tolist():
        if not seen[node]:
            seen[node] = True
            node_order.append(int(node))

    for _ in range(depth):
        next_frontier: list[int] = []
        for node in frontier.tolist():
            neigh = in_neighbors[int(node)]
            if fanout > 0 and neigh.shape[0] > fanout:
                sampled_neigh = rng.choice(neigh, size=fanout, replace=False)
            else:
                sampled_neigh = neigh
            for src in sampled_neigh.tolist():
                if not seen[src]:
                    seen[src] = True
                    node_order.append(int(src))
                    next_frontier.append(int(src))
                    if len(node_order) >= max_nodes:
                        next_frontier = []
                        break
            if len(node_order) >= max_nodes:
                break
        if not next_frontier:
            break
        frontier = np.asarray(next_frontier, dtype=np.int32)

    sub_nodes = np.asarray(node_order, dtype=np.int32)
    local_map = {int(n): i for i, n in enumerate(sub_nodes.tolist())}
    in_sub = seen
    edge_mask = in_sub[senders] & in_sub[receivers]
    sub_senders_global = senders[edge_mask]
    sub_receivers_global = receivers[edge_mask]

    sub_senders = np.asarray([local_map[int(x)] for x in sub_senders_global.tolist()], dtype=np.int32)
    sub_receivers = np.asarray([local_map[int(x)] for x in sub_receivers_global.tolist()], dtype=np.int32)
    local_seed = np.asarray([local_map[int(x)] for x in seed_nodes.tolist()], dtype=np.int32)
    return sub_nodes, sub_senders, sub_receivers, local_seed


def main() -> None:
    args = parse_args()
    if args.fanout < 1:
        raise ValueError("--fanout must be >= 1")
    if args.max_subgraph_nodes < args.batch_size:
        raise ValueError("--max-subgraph-nodes must be >= --batch-size")
    if args.early_stop_patience < 1:
        raise ValueError("--early-stop-patience must be >= 1")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_data(args.dataset_root)

    import keras
    import tensorflow as tf
    from ogb.nodeproppred import Evaluator

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    np.random.seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    evaluator = Evaluator(name="ogbn-arxiv")

    node_feat = data["node_feat"]
    senders = data["senders"]
    receivers = data["receivers"]
    n_nodes = data["n_nodes"]
    n_edges = data["n_edges"]
    in_neighbors = build_in_neighbors(n_nodes, senders, receivers)

    full_td = {
        "nodes": keras.ops.convert_to_tensor(node_feat, dtype="float32"),
        "edges": keras.ops.zeros((n_edges, 1), dtype="float32"),
        "senders": keras.ops.convert_to_tensor(senders, dtype="int32"),
        "receivers": keras.ops.convert_to_tensor(receivers, dtype="int32"),
        "n_edges": keras.ops.convert_to_tensor([n_edges], dtype="int32"),
        "n_nodes": keras.ops.convert_to_tensor([n_nodes], dtype="int32"),
        "n_graphs": keras.ops.convert_to_tensor(1, dtype="int32"),
        "global_attr": None,
        "global_reps_for_edges": keras.ops.zeros((n_edges,), dtype="int32"),
        "global_reps_for_nodes": keras.ops.zeros((n_nodes,), dtype="int32"),
    }

    y = data["y"]
    train_idx = data["train_idx"]
    valid_idx = data["valid_idx"]
    test_idx = data["test_idx"]
    num_classes = data["num_classes"]

    sampler = optuna.samplers.TPESampler(seed=args.seed) if args.sampler == "tpe" else optuna.samplers.RandomSampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=args.early_stop_patience)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_categorical("lr", [1e-4, 5e-4, 1e-3])
        depth = trial.suggest_categorical("depth", [3, 5, 7])
        width = trial.suggest_categorical("width", [256, 512])
        norm = trial.suggest_categorical("norm", ["batchnorm", "layernorm", "none"])
        dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.5])

        model = make_model(num_classes, depth, width, dropout, norm)
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        best_val = 0.0
        best_test = 0.0
        best_step = 0
        metric_rows: list[dict] = []
        rng = np.random.default_rng(args.seed + trial.number)

        for step in range(1, args.steps + 1):
            replace = args.batch_size > train_idx.shape[0]
            seed_nodes = rng.choice(train_idx, size=args.batch_size, replace=replace).astype(np.int32)
            seed_nodes = np.unique(seed_nodes)
            sub_nodes, sub_senders, sub_receivers, local_seed = sample_khop_subgraph(
                seed_nodes=seed_nodes,
                depth=depth,
                in_neighbors=in_neighbors,
                senders=senders,
                receivers=receivers,
                fanout=args.fanout,
                max_nodes=args.max_subgraph_nodes,
                rng=rng,
            )

            sub_n_nodes = int(sub_nodes.shape[0])
            sub_n_edges = int(sub_senders.shape[0])
            sub_td = {
                "nodes": keras.ops.convert_to_tensor(node_feat[sub_nodes], dtype="float32"),
                "edges": keras.ops.zeros((sub_n_edges, 1), dtype="float32"),
                "senders": keras.ops.convert_to_tensor(sub_senders, dtype="int32"),
                "receivers": keras.ops.convert_to_tensor(sub_receivers, dtype="int32"),
                "n_edges": keras.ops.convert_to_tensor([sub_n_edges], dtype="int32"),
                "n_nodes": keras.ops.convert_to_tensor([sub_n_nodes], dtype="int32"),
                "n_graphs": keras.ops.convert_to_tensor(1, dtype="int32"),
                "global_attr": None,
                "global_reps_for_edges": keras.ops.zeros((sub_n_edges,), dtype="int32"),
                "global_reps_for_nodes": keras.ops.zeros((sub_n_nodes,), dtype="int32"),
            }
            local_seed_t = keras.ops.convert_to_tensor(local_seed, dtype="int32")
            seed_labels_t = keras.ops.convert_to_tensor(y[seed_nodes], dtype="int32")

            with tf.GradientTape() as tape:
                logits = model(sub_td, training=True)
                seed_logits = keras.ops.take(logits, local_seed_t, axis=0)
                loss = loss_fn(seed_labels_t, seed_logits)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply(grads, model.trainable_weights)

            if step % args.eval_every == 0 or step == 1:
                logits_np = keras.ops.convert_to_numpy(model(full_td, training=False))
                train_acc = eval_acc(evaluator, logits_np, y, train_idx)
                val_acc = eval_acc(evaluator, logits_np, y, valid_idx)
                test_acc = eval_acc(evaluator, logits_np, y, test_idx)

                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc
                    best_step = step

                row = {
                    "step": step,
                    "loss": float(keras.ops.convert_to_numpy(loss)),
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "best_val": best_val,
                    "best_test": best_test,
                }
                metric_rows.append(row)
                trial.report(best_val, step)
                if trial.should_prune():
                    break

            if step - best_step >= args.early_stop_patience:
                break

        trial_dir = out_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        write_trial_metrics_csv(trial_dir / "metrics.csv", metric_rows)

        with (trial_dir / "params.json").open("w") as f:
            json.dump({"lr": lr, "depth": depth, "width": width, "norm": norm, "dropout": dropout}, f, indent=2)

        trial.set_user_attr("best_test", best_test)
        trial.set_user_attr("best_val", best_val)
        return best_val

    print(
        f"Starting Optuna study='{args.study_name}' trials={args.trials} "
        f"steps={args.steps} eval_every={args.eval_every} "
        f"batch_size={args.batch_size} fanout={args.fanout} max_nodes={args.max_subgraph_nodes} "
        f"patience={args.early_stop_patience}"
    )
    study.optimize(
        objective,
        n_trials=args.trials,
        catch=(tf.errors.ResourceExhaustedError,),
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials. Check OOMs or reduce model size/steps.")
        return

    print("Best trial:")
    print(f"  number: {study.best_trial.number}")
    print(f"  value (best val): {study.best_value:.6f}")
    print(f"  params: {study.best_trial.params}")
    print(f"  best_test: {study.best_trial.user_attrs.get('best_test')}")


if __name__ == "__main__":
    main()
