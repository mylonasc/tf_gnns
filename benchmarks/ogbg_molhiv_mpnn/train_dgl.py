"""DGL MPNN benchmark implementation for ogbg-molhiv."""

from __future__ import annotations

import dgl
import dgl.nn as dglnn
import torch
import torch.nn.functional as F
from torch import nn

from .common import now_ms, summarize_times


class DGLMPNN(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.edge_net1 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * in_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * in_dim, hidden_dim * in_dim),
        )
        self.edge_net2 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim),
        )
        self.conv1 = dglnn.NNConv(in_dim, hidden_dim, self.edge_net1, aggregator_type="mean")
        self.conv2 = dglnn.NNConv(hidden_dim, hidden_dim, self.edge_net2, aggregator_type="mean")
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, graph):
        x = graph.ndata["x"]
        e = graph.edata["edge_attr"]
        x = F.relu(self.conv1(graph, x, e))
        x = F.relu(self.conv2(graph, x, e))
        graph.ndata["h"] = x
        pooled = dgl.mean_nodes(graph, "h")
        return self.head(pooled)


def _benchmark_impl(dgl_samples, hidden_dim, lr, warmup_steps, bench_steps, sample_mode, compile_model):
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
    edge_dim = int(dgl_samples[0].edata["edge_attr"].shape[1])
    model = DGLMPNN(in_dim=in_dim, edge_dim=edge_dim, hidden_dim=hidden_dim).to(device)
    if compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this torch build")
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

    mode_prefix = "torch_compile" if compile_model else "torch_eager"
    mode = f"{mode_prefix}_cuda" if device.type == "cuda" else f"{mode_prefix}_cpu"
    if not dgl_cuda_enabled:
        mode += "_dgl_cpu_build"
    metrics = summarize_times(step_times)
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


def benchmark_dgl(dgl_samples, hidden_dim, lr, warmup_steps, bench_steps, sample_mode="single"):
    return _benchmark_impl(dgl_samples, hidden_dim, lr, warmup_steps, bench_steps, sample_mode, False)


def benchmark_dgl_compile(dgl_samples, hidden_dim, lr, warmup_steps, bench_steps, sample_mode="single"):
    return _benchmark_impl(dgl_samples, hidden_dim, lr, warmup_steps, bench_steps, sample_mode, True)
