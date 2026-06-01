"""DGL benchmark implementation for ogbg-molhiv."""

from __future__ import annotations

import dgl
import dgl.nn as dglnn
import torch
import torch.nn.functional as F
from torch import nn

from .common import now_ms, summarize_times


class DGLGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, norm="both")
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, norm="both")
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, graph):
        x = graph.ndata["x"]
        x = F.relu(self.conv1(graph, x))
        x = F.relu(self.conv2(graph, x))
        graph.ndata["h"] = x
        pooled = dgl.mean_nodes(graph, "h")
        return self.head(pooled)


def benchmark_dgl(
    dgl_samples,
    hidden_dim: int,
    lr: float,
    warmup_steps: int,
    bench_steps: int,
    sample_mode: str = "single",
):
    return _benchmark_dgl_impl(
        dgl_samples,
        hidden_dim=hidden_dim,
        lr=lr,
        warmup_steps=warmup_steps,
        bench_steps=bench_steps,
        sample_mode=sample_mode,
        compile_model=False,
    )


def _benchmark_dgl_impl(
    dgl_samples,
    hidden_dim: int,
    lr: float,
    warmup_steps: int,
    bench_steps: int,
    sample_mode: str,
    compile_model: bool,
):
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dgl_cuda_enabled = True
    if device.type == "cuda":
        try:
            _ = dgl.graph(([0], [0]), num_nodes=1).to(device)
        except Exception:
            dgl_cuda_enabled = False
            device = torch.device("cpu")

    in_dim = int(dgl_samples[0].ndata["x"].shape[1])
    model = DGLGCN(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    if compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this torch build")
        try:
            from torch import _dynamo as dynamo

            dynamo.config.capture_scalar_outputs = True
        except Exception:
            pass
        model = torch.compile(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(dgl_samples)

    def pick(i):
        if sample_mode == "single":
            return dgl_samples[0]
        return dgl_samples[i % n]

    def step(g):
        bg = dgl.batch([g]).to(device)
        y = g.y.view(-1, 1).to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(bg)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

    for i in range(warmup_steps):
        step(pick(i))

    step_times = []
    for i in range(bench_steps):
        t0 = now_ms()
        step(pick(i))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = now_ms()
        step_times.append(t1 - t0)

    metrics = summarize_times(step_times)
    mode_prefix = "torch_compile" if compile_model else "torch_eager"
    mode = f"{mode_prefix}_cuda" if device.type == "cuda" else f"{mode_prefix}_cpu"
    if not dgl_cuda_enabled:
        mode += "_dgl_cpu_build"
        print(
            "[dgl] Warning: CUDA-capable torch detected but DGL CUDA backend is unavailable; "
            "running DGL benchmark on CPU."
        )
    metrics.update(
        {
            "framework": "dgl",
            "mode": mode,
            "device": device.type,
            "feature_dtype": str(dgl_samples[0].ndata["x"].dtype),
            "index_dtype": str(dgl_samples[0].edges()[0].dtype),
            "param_dtype": str(next(model.parameters()).dtype),
        }
    )
    return metrics


def benchmark_dgl_compile(
    dgl_samples,
    hidden_dim: int,
    lr: float,
    warmup_steps: int,
    bench_steps: int,
    sample_mode: str = "single",
):
    return _benchmark_dgl_impl(
        dgl_samples,
        hidden_dim=hidden_dim,
        lr=lr,
        warmup_steps=warmup_steps,
        bench_steps=bench_steps,
        sample_mode=sample_mode,
        compile_model=True,
    )
