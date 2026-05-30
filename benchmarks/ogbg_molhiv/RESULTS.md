# OGBG-MOLHIV Benchmark Snapshot

## Run configuration

- Dataset: `ogbg-molhiv`
- Steps: `10000`
- Warmup: `100`
- Hidden dim: `128`
- Max graphs loaded: `8`
- Sample mode: `single`
- Command:

```bash
scripts/run_with_tf_gpu uv run python benchmarks/ogbg_molhiv/run_suite.py --steps 10000 --warmup 100 --hidden-dim 128 --max-graphs 8 --sample-mode single
```

## Results

| framework | mode | device | avg_step_ms | std_step_ms | steps_per_sec |
| --- | --- | --- | ---: | ---: | ---: |
| tf_gnns | tf_function | gpu | 0.744003 | 0.099687 | 1344.080613 |
| pyg | torch_eager | cuda | 1.274210 | 0.075739 | 784.800079 |
| pyg | torch_compile | cuda | 1.581687 | 0.100324 | 632.236347 |
| dgl | torch_eager_cpu_dgl_cpu_build | cpu | 2.709928 | 0.132478 | 369.013442 |
| dgl | torch_compile_cpu_dgl_cpu_build | cpu | 5.282029 | 0.175837 | 189.321192 |

## Notes

- The DGL rows are CPU-only (`dgl_cpu_build`), so they are not directly comparable with GPU rows.
- In this run, PyG eager is faster than PyG compiled. This is likely due to compile graph breaks and overhead in this small single-graph microbenchmark regime.
- To improve fairness in follow-up comparisons:
  - Ensure DGL uses a CUDA-enabled build.
  - Add larger batched workloads and static-shape runs for `torch.compile`.
  - Add repeated benchmark runs and confidence intervals.
