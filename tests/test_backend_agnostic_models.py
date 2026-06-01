import importlib.util
import json
import os
import subprocess
import sys

import pytest


def _module_available(name):
    return importlib.util.find_spec(name) is not None


def _backend_ready(backend):
    if backend == "jax":
        return _module_available("jax")
    if backend == "torch":
        return _module_available("torch")
    if backend == "tensorflow":
        return _module_available("tensorflow")
    return False


@pytest.mark.parametrize(
    ("backend", "mode"),
    [
        ("tensorflow", "eager"),
        ("tensorflow", "compiled"),
        ("jax", "eager"),
        ("jax", "compiled"),
        ("torch", "eager"),
    ],
)
@pytest.mark.parametrize("model_kind", ["mpnn", "gcn"])
def test_tf_gnns_models_train_step_smoke_across_backends(backend, mode, model_kind):
    if not _backend_ready(backend):
        pytest.skip(f"{backend} is not installed in this environment")

    code = r'''
import json
import math
import os

os.environ["KERAS_BACKEND"] = os.environ["BACKEND"]

import keras

from tf_gnns.models.gcn import GCNv2
from tf_gnns.models.graphnet import GraphNetMPNN_MLP


def make_graph_td():
    nodes = keras.ops.convert_to_tensor([[1.0, 0.0], [2.0, -1.0], [3.0, 1.0]], dtype="float32")
    edges = keras.ops.convert_to_tensor([[0.5, -1.0], [1.5, 2.0], [-0.3, 0.7]], dtype="float32")
    senders = keras.ops.convert_to_tensor([0, 1, 2], dtype="int32")
    receivers = keras.ops.convert_to_tensor([1, 2, 0], dtype="int32")
    return {
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "n_edges": keras.ops.convert_to_tensor([3], dtype="int32"),
        "n_nodes": keras.ops.convert_to_tensor([3], dtype="int32"),
        "n_graphs": keras.ops.convert_to_tensor(1, dtype="int32"),
        "global_attr": None,
        "global_reps_for_edges": keras.ops.convert_to_tensor([0, 0, 0], dtype="int32"),
        "global_reps_for_nodes": keras.ops.convert_to_tensor([0, 0, 0], dtype="int32"),
    }


class TinyMPNN(keras.Model):
    def __init__(self):
        super().__init__()
        self.core = GraphNetMPNN_MLP(units=8, core_units=8, core_steps=1)
        self.head = keras.layers.Dense(1, dtype="float32")

    def call(self, td):
        td = self.core(td)
        pooled = keras.ops.mean(td["nodes"], axis=0, keepdims=True)
        return self.head(pooled)


class TinyGCN(keras.Model):
    def __init__(self):
        super().__init__()
        self.core = GCNv2(
            hidden_units=8,
            output_units=4,
            num_layers=3,
            input_dropout_rate=0.0,
            dropout_rate=0.0,
            batchnorm=False,
            layernorm=False,
            residual=True,
            residual_projection=True,
        )
        self.head = keras.layers.Dense(1, dtype="float32")

    def call(self, td):
        td = self.core(td)
        pooled = keras.ops.mean(td["nodes"], axis=0, keepdims=True)
        return self.head(pooled)


td = make_graph_td()
y = keras.ops.convert_to_tensor([[1.0]], dtype="float32")

if os.environ["MODEL_KIND"] == "gcn":
    model = TinyGCN()
else:
    model = TinyMPNN()

_ = model(td)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.MeanSquaredError(),
    run_eagerly=(os.environ["MODE"] == "eager"),
    jit_compile=(os.environ["MODE"] == "compiled"),
)

loss = model.train_on_batch(td, y)
loss_val = float(keras.ops.convert_to_numpy(loss))
pred = model(td)
pred_val = float(keras.ops.convert_to_numpy(keras.ops.mean(pred)))
if not math.isfinite(loss_val):
    raise RuntimeError(f"Non-finite loss: {loss_val}")
if not math.isfinite(pred_val):
    raise RuntimeError(f"Non-finite prediction: {pred_val}")
print(json.dumps({
    "backend": os.environ["BACKEND"],
    "mode": os.environ["MODE"],
    "model_kind": os.environ["MODEL_KIND"],
    "loss": loss_val,
    "pred": pred_val,
}))
'''

    env = dict(os.environ)
    env["BACKEND"] = backend
    env["MODE"] = mode
    env["MODEL_KIND"] = model_kind
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["backend"] == backend
    assert payload["mode"] == mode
    assert payload["model_kind"] == model_kind
    assert isinstance(payload["loss"], float)
    assert isinstance(payload["pred"], float)
