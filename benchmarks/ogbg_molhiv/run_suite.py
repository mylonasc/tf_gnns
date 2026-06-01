"""Run tf_gnns vs PyG vs DGL benchmark on ogbg-molhiv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import urllib.request

from benchmarks.ogbg_molhiv.common import BenchConfig, set_all_seeds
from benchmarks.ogbg_molhiv.data import (
    load_ogbg_molhiv,
    make_framework_samples,
    validate_dataset_indexed_graphs,
    validate_framework_sample_alignment,
)


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark GCN frameworks on ogbg-molhiv")
    p.add_argument("--dataset-root", type=str, default="dataset")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-graphs", type=int, default=256)
    p.add_argument(
        "--sample-mode",
        type=str,
        choices=["single", "cycle"],
        default="single",
        help="single=reuse one graph to avoid retracing/recompiles; cycle=iterate over sampled graphs",
    )
    p.add_argument(
        "--tfgnns-backends",
        type=str,
        default="tensorflow,torch,jax",
        help="Comma-separated Keras backends for tf_gnns benchmark",
    )
    p.add_argument(
        "--tfgnns-modes",
        type=str,
        default="eager,compiled",
        help="Comma-separated tf_gnns benchmark modes: eager,compiled",
    )
    p.add_argument("--feature-dtype", type=str, default="float32")
    p.add_argument("--index-dtype", type=str, default="auto")
    p.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/ogbg_molhiv_results.csv",
    )
    p.add_argument(
        "--include-dgl",
        action="store_true",
        help="Include DGL benchmark runs (disabled by default).",
    )
    return p.parse_args()


def _url_exists(url: str, timeout: int = 6) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def _dgl_compatibility_warning() -> None:
    """Print a warning if local Torch/CUDA appears unsupported by DGL wheels."""
    try:
        import torch
    except Exception:
        print("[compat] Torch not importable; skipping DGL compatibility check.")
        return

    torch_base = torch.__version__.split("+")[0]
    m = re.match(r"(\d+)\.(\d+)", torch_base)
    if not m:
        print(f"[compat] Could not parse torch version: {torch.__version__}")
        return
    torch_mm = f"{m.group(1)}.{m.group(2)}"

    cuda = getattr(torch.version, "cuda", None)
    cuda_tag = None
    if cuda:
        cm = re.match(r"(\d+)\.(\d+)", cuda)
        if cm:
            cuda_tag = f"cu{cm.group(1)}{cm.group(2)}"

    base = f"https://data.dgl.ai/wheels/torch-{torch_mm}"
    urls = []
    if cuda_tag:
        urls.append(f"{base}/{cuda_tag}/repo.html")
    urls.append(f"{base}/repo.html")
    urls.append(f"{base}/cpu/repo.html")

    ok = any(_url_exists(u) for u in urls)
    if ok:
        print(
            f"[compat] DGL wheel index found for torch-{torch_mm}"
            + (f" ({cuda_tag})" if cuda_tag else "")
            + "."
        )
        return

    print(
        "[compat] Warning: No direct DGL wheel index found for "
        f"torch-{torch_mm}" + (f"/{cuda_tag}" if cuda_tag else "") + "."
    )
    print(
        "[compat] DGL may not support your latest Torch/CUDA combo out of the box. "
        "Try `uv run python scripts/install_dgl.py` for fallback probing."
    )


def main():
    args = parse_args()

    if args.include_dgl:
        _dgl_compatibility_warning()

    import pandas as pd

    from benchmarks.ogbg_molhiv.train_pyg import benchmark_pyg, benchmark_pyg_compile
    from benchmarks.ogbg_molhiv.train_tf_gnns import benchmark_tf_gnns

    benchmark_dgl = None
    benchmark_dgl_compile = None
    if args.include_dgl:
        try:
            from benchmarks.ogbg_molhiv.train_dgl import benchmark_dgl, benchmark_dgl_compile
        except Exception as exc:
            raise RuntimeError(
                "DGL benchmark import failed. Install DGL manually for your platform, "
                "e.g. `uv pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/repo.html`."
            ) from exc

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
        dataset,
        splits.train_idx,
        max_graphs=args.max_graphs,
        include_dgl=args.include_dgl,
    )
    validate_framework_sample_alignment(tf_samples, pyg_samples, dgl_samples)

    results = []
    tfgnns_backends = [b.strip() for b in args.tfgnns_backends.split(",") if b.strip()]
    tfgnns_modes = [m.strip() for m in args.tfgnns_modes.split(",") if m.strip()]

    for backend_name in tfgnns_backends:
        for mode in tfgnns_modes:
            if backend_name == "tensorflow":
                use_tf_function = mode == "compiled"
                try:
                    result = benchmark_tf_gnns(
                        tf_samples,
                        hidden_dim=cfg.hidden_dim,
                        lr=cfg.learning_rate,
                        warmup_steps=cfg.warmup_steps,
                        bench_steps=cfg.bench_steps,
                        sample_mode=args.sample_mode,
                        use_tf_function=use_tf_function,
                        train_step_jit_compile=False,
                    )
                    result["backend"] = "tensorflow"
                    result["mode"] = "keras_compiled" if use_tf_function else "keras_eager"
                    result["feature_dtype"] = str(tf_samples[0]["nodes"].dtype)
                    result["index_dtype"] = str(tf_samples[0]["senders"].dtype)
                    result["param_dtype"] = str(result.get("param_dtype", "float32"))
                    results.append(result)
                except Exception as exc:
                    print(f"[tf_gnns] backend=tensorflow mode={mode} skipped: {exc}")
                continue

            cmd = [
                "uv",
                "run",
                "python",
                "benchmarks/ogbg_molhiv/run_tfgnns_backend.py",
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
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(
                    f"[tf_gnns] backend={backend_name} mode={mode} skipped:\n"
                    f"{proc.stderr.strip() or proc.stdout.strip()}"
                )
                continue
            try:
                result_line = proc.stdout.strip().splitlines()[-1]
                results.append(json.loads(result_line))
            except Exception as exc:
                print(
                    f"[tf_gnns] backend={backend_name} mode={mode} parse failed: {exc}"
                )
    results.append(
        benchmark_pyg(
            pyg_samples,
            hidden_dim=cfg.hidden_dim,
            lr=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            bench_steps=cfg.bench_steps,
            sample_mode=args.sample_mode,
        )
    )
    try:
        results.append(
            benchmark_pyg_compile(
                pyg_samples,
                hidden_dim=cfg.hidden_dim,
                lr=cfg.learning_rate,
                warmup_steps=cfg.warmup_steps,
                bench_steps=cfg.bench_steps,
                sample_mode=args.sample_mode,
            )
        )
    except Exception as exc:
        print(f"[pyg] compile benchmark skipped: {exc}")

    if args.include_dgl:
        results.append(
            benchmark_dgl(
                dgl_samples,
                hidden_dim=cfg.hidden_dim,
                lr=cfg.learning_rate,
                warmup_steps=cfg.warmup_steps,
                bench_steps=cfg.bench_steps,
                sample_mode=args.sample_mode,
            )
        )
        try:
            results.append(
                benchmark_dgl_compile(
                    dgl_samples,
                    hidden_dim=cfg.hidden_dim,
                    lr=cfg.learning_rate,
                    warmup_steps=cfg.warmup_steps,
                    bench_steps=cfg.bench_steps,
                    sample_mode=args.sample_mode,
                )
            )
        except Exception as exc:
            print(f"[dgl] compile benchmark skipped: {exc}")
    else:
        print("[suite] DGL benchmarks disabled by default. Use --include-dgl to enable.")

    df = pd.DataFrame(results)
    df.insert(0, "dataset", "ogbg-molhiv")
    df.insert(1, "steps", cfg.bench_steps)
    df.insert(2, "warmup", cfg.warmup_steps)
    df.insert(3, "hidden_dim", cfg.hidden_dim)
    df.insert(4, "max_graphs", args.max_graphs)
    df.insert(5, "sample_mode", args.sample_mode)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
