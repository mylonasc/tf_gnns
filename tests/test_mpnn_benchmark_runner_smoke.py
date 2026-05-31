import json
import os
import subprocess
import sys

import pytest


def _module_available(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


@pytest.mark.parametrize(("backend", "mode"), [("tensorflow", "eager"), ("jax", "eager")])
def test_mpnn_benchmark_backend_runner_emits_result_json(backend, mode):
    if backend == "jax" and not _module_available("jax"):
        pytest.skip("JAX is not installed in this environment")
    code = r'''
import json
import os
import runpy
import sys
import types
import numpy as np

os.environ["KERAS_BACKEND"] = os.environ["BACKEND"]


class _FakeDataset:
    def __init__(self, name, root):
        self.name = name
        self.root = root

    def get_idx_split(self):
        return {"train": np.array([0], dtype=np.int64)}

    def __getitem__(self, idx):
        graph = {
            "edge_index": np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64),
            "node_feat": np.array([[1.0, 0.0], [2.0, -1.0], [3.0, 1.0]], dtype=np.float32),
            "edge_feat": np.array([[0.5, -1.0], [1.5, 2.0], [-0.3, 0.7]], dtype=np.float32),
            "num_nodes": 3,
        }
        y = np.array([1.0], dtype=np.float32)
        return graph, y


ogb_mod = types.ModuleType("ogb")
graphproppred_mod = types.ModuleType("ogb.graphproppred")
graphproppred_mod.GraphPropPredDataset = _FakeDataset
ogb_mod.graphproppred = graphproppred_mod
sys.modules["ogb"] = ogb_mod
sys.modules["ogb.graphproppred"] = graphproppred_mod

sys.argv = [
    "run_tfgnns_backend.py",
    "--backend", os.environ["BACKEND"],
    "--mode", os.environ["MODE"],
    "--steps", "1",
    "--warmup", "0",
    "--max-graphs", "1",
    "--feature-dtype", "float32",
    "--index-dtype", "int32",
]

runpy.run_path("benchmarks/ogbg_molhiv_mpnn/run_tfgnns_backend.py", run_name="__main__")
'''

    env = dict(os.environ)
    env["BACKEND"] = backend
    env["MODE"] = mode
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    for key in [
        "framework",
        "mode",
        "backend",
        "avg_step_ms",
        "std_step_ms",
        "steps_per_sec",
    ]:
        assert key in payload
    assert payload["framework"] == "tf_gnns"
    assert payload["backend"] == backend
