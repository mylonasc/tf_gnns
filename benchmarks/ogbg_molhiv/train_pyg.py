"""PyG benchmark implementation for ogbg-molhiv."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch

from .common import now_ms, summarize_times


class PyGGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # We benchmark one graph per step for strict parity with tf_gnns path.
        # Avoids graph breaks in torch.compile from dynamic scatter internals.
        x = x.mean(dim=0, keepdim=True)
        return self.head(x)


def benchmark_pyg(
    pyg_samples,
    hidden_dim: int,
    lr: float,
    warmup_steps: int,
    bench_steps: int,
    sample_mode: str = "single",
):
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = int(pyg_samples[0].x.shape[1])
    model = PyGGCN(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
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
    metrics.update({"framework": "pyg", "mode": "torch_eager", "device": device.type})
    return metrics


def benchmark_pyg_compile(
    pyg_samples,
    hidden_dim: int,
    lr: float,
    warmup_steps: int,
    bench_steps: int,
    sample_mode: str = "single",
):
    torch.set_float32_matmul_precision("high")
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this torch build")
    try:
        from torch import _dynamo as dynamo

        dynamo.config.capture_scalar_outputs = True
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = int(pyg_samples[0].x.shape[1])
    model = PyGGCN(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
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
        {"framework": "pyg", "mode": "torch_compile", "device": device.type}
    )
    return metrics
