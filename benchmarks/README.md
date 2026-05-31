# OGBG-MOLHIV Benchmark Suites

These suites benchmark matched training-step performance across:

- `tf_gnns` (Keras multi-backend paths)
- PyG (`torch_geometric`)
- DGL (`dgl`)

Default metric is average wall-clock time for 100 gradient steps.

The results table includes:
- `framework`
- `mode` (eager/compile variants)
- `device` (`cpu` or `cuda`/`gpu`)
- `feature_dtype`, `index_dtype`, `param_dtype`
- `avg_step_ms`, `std_step_ms`, `steps_per_sec`

## Setup

```bash
uv sync --group dev --group bench
```

If your system needs CUDA-specific wheels, install framework packages according
to your platform guidance before running this suite.

### DGL install (manual)

`dgl` is intentionally not pinned in `bench` dependencies because wheel
availability varies by OS, Python, CUDA, and PyTorch build.
Install DGL manually from the official wheel index matching your setup, for
example:

```bash
uv pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/repo.html
```

For CPU-only installs, use the CPU wheel index from DGL docs.

## Run (GCN)

```bash
uv run python benchmarks/ogbg_molhiv/run_suite.py --steps 100 --warmup 20 --batch-size 64 --sample-mode single --tfgnns-backends tensorflow,torch,jax --tfgnns-modes eager,compiled --feature-dtype float32 --index-dtype int32
```

Results are written to `benchmarks/results/ogbg_molhiv_results.csv`.

## Run (MPNN/GraphNet)

```bash
uv run python benchmarks/ogbg_molhiv_mpnn/run_suite.py --steps 100 --warmup 20 --batch-size 64 --sample-mode single --tfgnns-backends tensorflow,torch,jax --tfgnns-modes eager,compiled --feature-dtype float32 --index-dtype auto
```

Results are written to `benchmarks/results/ogbg_molhiv_mpnn_results.csv`.

## Notes on parity

- Benchmarks use node features only (edge features are ignored) for strict
  vanilla GCN parity.
- All frameworks run the same high-level architecture:
  `GCN -> ReLU -> GCN -> ReLU -> GlobalMeanPool -> Linear`.
- The suite validates dataset conversion consistency before timing.
- For fair compile comparisons, prefer `--sample-mode single` to avoid
  shape-driven retracing/recompilation noise.
- Compare rows with the same `device` type. If DGL mode ends with
  `_dgl_cpu_build`, that row is CPU-only and should not be compared to GPU rows.
- The MPNN suite uses edge features in all frameworks. The GCN suite ignores edge
  features for strict vanilla GCN parity.
- The `GraphNetMPNN_MLP` path accepts `tensorflow,torch,jax` backend selections.
- If a backend/mode pair fails at runtime, the suite logs the subprocess error
  and continues with other frameworks.
