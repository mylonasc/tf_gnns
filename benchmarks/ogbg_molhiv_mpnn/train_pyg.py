"""PyG MPNN benchmark implementation for ogbg-molhiv."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import NNConv

from .common import now_ms, summarize_times


class PyGMPNN(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.edge_mlp1 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * in_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * in_dim, hidden_dim * in_dim),
        )
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim),
        )
        self.conv1 = NNConv(in_dim, hidden_dim, self.edge_mlp1, aggr="mean")
        self.conv2 = NNConv(hidden_dim, hidden_dim, self.edge_mlp2, aggr="mean")
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = x.mean(dim=0, keepdim=True)
        return self.head(x)


def _benchmark_impl(pyg_samples, hidden_dim, lr, warmup_steps, bench_steps, sample_mode, compile_model):
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = int(pyg_samples[0].x.shape[1])
    edge_dim = int(pyg_samples[0].edge_attr.shape[1])
    model = PyGMPNN(in_dim=in_dim, edge_dim=edge_dim, hidden_dim=hidden_dim).to(device)
    if compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this torch build")
        model = torch.compile(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(pyg_samples)

    def pick(i):
        if sample_mode == "single":
            return pyg_samples[0]
        return pyg_samples[i % n]

    def step(data):
        batch = Batch.from_data_list([data]).to(device)
        y = batch.y.view(-1, 1)
        opt.zero_grad(set_to_none=True)
        logits = model(batch)
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
    metrics.update(
        {
            "framework": "pyg",
            "mode": "torch_compile" if compile_model else "torch_eager",
            "device": device.type,
            "feature_dtype": str(pyg_samples[0].x.dtype),
            "index_dtype": str(pyg_samples[0].edge_index.dtype),
            "param_dtype": str(next(model.parameters()).dtype),
        }
    )
    return metrics


def benchmark_pyg(pyg_samples, hidden_dim, lr, warmup_steps, bench_steps, sample_mode="single"):
    return _benchmark_impl(pyg_samples, hidden_dim, lr, warmup_steps, bench_steps, sample_mode, False)


def benchmark_pyg_compile(pyg_samples, hidden_dim, lr, warmup_steps, bench_steps, sample_mode="single"):
    return _benchmark_impl(pyg_samples, hidden_dim, lr, warmup_steps, bench_steps, sample_mode, True)
