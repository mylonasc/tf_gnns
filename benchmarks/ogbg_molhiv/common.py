"""Shared utilities for OGBG-MOLHIV benchmarking."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class BenchConfig:
    batch_size: int = 64
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    warmup_steps: int = 20
    bench_steps: int = 100
    seed: int = 42


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def now_ms() -> float:
    return time.perf_counter() * 1e3


def summarize_times(step_times_ms: list[float]) -> dict:
    arr = np.array(step_times_ms, dtype=np.float64)
    return {
        "avg_step_ms": float(arr.mean()),
        "std_step_ms": float(arr.std()),
        "steps_per_sec": float(1000.0 / arr.mean()),
    }
