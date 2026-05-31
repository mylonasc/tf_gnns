"""Run tf_gnns GraphNetMPNN vs PyG vs DGL benchmark on ogbg-molhiv."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from benchmarks.ogbg_molhiv.common import BenchConfig, set_all_seeds
from benchmarks.ogbg_molhiv.run_suite import _dgl_compatibility_warning
from benchmarks.ogbg_molhiv_mpnn.data import (
    load_ogbg_molhiv,
    make_framework_samples,
    validate_dataset_indexed_graphs,
    validate_framework_sample_alignment,
)


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark MPNN frameworks on ogbg-molhiv")
    p.add_argument("--dataset-root", type=str, default="dataset")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-graphs", type=int, default=256)
    p.add_argument("--sample-mode", type=str, choices=["single", "cycle"], default="single")
    p.add_argument("--tfgnns-backends", type=str, default="tensorflow,torch,jax")
    p.add_argument("--tfgnns-modes", type=str, default="eager,compiled")
    p.add_argument("--feature-dtype", type=str, default="float32")
    p.add_argument("--index-dtype", type=str, default="auto")
    p.add_argument("--output", type=str, default="benchmarks/results/ogbg_molhiv_mpnn_results.csv")
    return p.parse_args()


def main():
    args = parse_args()
    _dgl_compatibility_warning()

    import pandas as pd

    from benchmarks.ogbg_molhiv_mpnn.train_dgl import benchmark_dgl, benchmark_dgl_compile
    from benchmarks.ogbg_molhiv_mpnn.train_pyg import benchmark_pyg, benchmark_pyg_compile

    cfg = BenchConfig(
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        bench_steps=args.steps,
        seed=args.seed,
    )
    set_all_seeds(cfg.seed)

    dataset, splits = load_ogbg_molhiv(root=args.dataset_root)
    validate_dataset_indexed_graphs(dataset, splits.train_idx)
    tf_samples, pyg_samples, dgl_samples = make_framework_samples(
        dataset, splits.train_idx, max_graphs=args.max_graphs, include_tf=False
    )
    validate_framework_sample_alignment(tf_samples, pyg_samples, dgl_samples)

    results = []
    tfgnns_backends = [b.strip() for b in args.tfgnns_backends.split(",") if b.strip()]
    tfgnns_modes = [m.strip() for m in args.tfgnns_modes.split(",") if m.strip()]

    for backend_name in tfgnns_backends:
        for mode in tfgnns_modes:
            print("\n" + "=" * 72)
            print(f"[suite] running tf_gnns mpnn backend={backend_name} mode={mode}")
            print("=" * 72)
            cmd = [
                "uv",
                "run",
                "python",
                "benchmarks/ogbg_molhiv_mpnn/run_tfgnns_backend.py",
                "--backend",
                backend_name,
                "--mode",
                mode,
                "--dataset-root",
                args.dataset_root,
                "--steps",
                str(cfg.bench_steps),
                "--warmup",
                str(cfg.warmup_steps),
                "--hidden-dim",
                str(cfg.hidden_dim),
                "--max-graphs",
                str(args.max_graphs),
                "--seed",
                str(cfg.seed),
                "--sample-mode",
                args.sample_mode,
                "--feature-dtype",
                args.feature_dtype,
                "--index-dtype",
                args.index_dtype,
            ]
            env = None
            if backend_name == "jax":
                env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            output_lines = []
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                output_lines.append(line.rstrip("\n"))
            return_code = proc.wait()
            if return_code != 0:
                details = "\n".join(output_lines).strip() or "(no output)"
                print(
                    f"[tf_gnns] backend={backend_name} mode={mode} skipped "
                    f"(exit={return_code}):\n{details}"
                )
                continue
            json_line = None
            for line in reversed(output_lines):
                l = line.strip()
                if l.startswith("{") and l.endswith("}"):
                    json_line = l
                    break
            if json_line is None:
                print(
                    f"[tf_gnns] backend={backend_name} mode={mode} skipped "
                    "(no JSON result line found)."
                )
                continue
            results.append(json.loads(json_line))

    print("\n" + "=" * 72)
    print("[suite] running PyG benchmarks")
    print("=" * 72)
    results.append(benchmark_pyg(pyg_samples, cfg.hidden_dim, cfg.learning_rate, cfg.warmup_steps, cfg.bench_steps, args.sample_mode))
    try:
        results.append(benchmark_pyg_compile(pyg_samples, cfg.hidden_dim, cfg.learning_rate, cfg.warmup_steps, cfg.bench_steps, args.sample_mode))
    except Exception as exc:
        print(f"[pyg] compile benchmark skipped: {exc}")

    print("\n" + "=" * 72)
    print("[suite] running DGL benchmarks")
    print("=" * 72)
    results.append(benchmark_dgl(dgl_samples, cfg.hidden_dim, cfg.learning_rate, cfg.warmup_steps, cfg.bench_steps, args.sample_mode))
    try:
        results.append(benchmark_dgl_compile(dgl_samples, cfg.hidden_dim, cfg.learning_rate, cfg.warmup_steps, cfg.bench_steps, args.sample_mode))
    except Exception as exc:
        print(f"[dgl] compile benchmark skipped: {exc}")

    df = pd.DataFrame(results)
    df.insert(0, "dataset", "ogbg-molhiv")
    df.insert(1, "model_family", "mpnn_graphnet")
    df.insert(2, "steps", cfg.bench_steps)
    df.insert(3, "warmup", cfg.warmup_steps)
    df.insert(4, "hidden_dim", cfg.hidden_dim)
    df.insert(5, "max_graphs", args.max_graphs)
    df.insert(6, "sample_mode", args.sample_mode)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
