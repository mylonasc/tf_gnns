import importlib.util
import json
import os
import subprocess
import sys

import pytest


def _module_available(name):
    return importlib.util.find_spec(name) is not None


@pytest.mark.parametrize(
    ("backend", "mode", "train_step"),
    [
        ("tensorflow", "eager", True),
        ("tensorflow", "compiled", True),
        ("jax", "eager", False),
        ("jax", "compiled", False),
    ],
)
def test_graphnet_mpnn_train_step_smoke_across_backends(backend, mode, train_step):
    if backend == "jax" and not _module_available("jax"):
        pytest.skip("JAX is not installed in this environment")

    code = r'''
import json
import math
import os

os.environ["KERAS_BACKEND"] = os.environ["BACKEND"]

import keras

from tf_gnns.models.graphnet import GraphNetMPNN_MLP


class TinyMPNN(keras.Model):
    def __init__(self):
        super().__init__()
        self.mpnn = GraphNetMPNN_MLP(units=8, core_units=8, core_steps=1)
        self.head = keras.layers.Dense(1, dtype="float32")

    def call(self, td):
        td = self.mpnn(td)
        pooled = keras.ops.mean(td["nodes"], axis=0, keepdims=True)
        return self.head(pooled)


nodes = keras.ops.convert_to_tensor([[1.0, 0.0], [2.0, -1.0], [3.0, 1.0]], dtype="float32")
edges = keras.ops.convert_to_tensor([[0.5, -1.0], [1.5, 2.0], [-0.3, 0.7]], dtype="float32")
senders = keras.ops.convert_to_tensor([0, 1, 2], dtype="int32")
receivers = keras.ops.convert_to_tensor([1, 2, 0], dtype="int32")
td = {
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
y = keras.ops.convert_to_tensor([[1.0]], dtype="float32")

model = TinyMPNN()
_ = model(td)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    run_eagerly=(os.environ["MODE"] == "eager"),
    jit_compile=(os.environ["MODE"] == "compiled"),
)
if os.environ["TRAIN_STEP"] == "1":
    loss = model.train_on_batch(td, y)
    loss_val = float(keras.ops.convert_to_numpy(loss))
else:
    out = model(td)
    loss_val = float(keras.ops.convert_to_numpy(keras.ops.mean(out)))
if not math.isfinite(loss_val):
    raise RuntimeError(f"Non-finite loss: {loss_val}")
print(json.dumps({"backend": os.environ["BACKEND"], "mode": os.environ["MODE"], "loss": loss_val}))
'''

    env = dict(os.environ)
    env["BACKEND"] = backend
    env["MODE"] = mode
    env["TRAIN_STEP"] = "1" if train_step else "0"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["backend"] == backend
    assert payload["mode"] == mode
    assert isinstance(payload["loss"], float)


@pytest.mark.xfail(reason="Keras JAX train_on_batch path not stable for GraphNetMPNN yet")
def test_graphnet_mpnn_jax_train_step_known_failure_contract():
    if not _module_available("jax"):
        pytest.skip("JAX is not installed in this environment")

    code = r'''
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
from tf_gnns.models.graphnet import GraphNetMPNN_MLP


class TinyMPNN(keras.Model):
    def __init__(self):
        super().__init__()
        self.mpnn = GraphNetMPNN_MLP(units=8, core_units=8, core_steps=1)
        self.head = keras.layers.Dense(1, dtype="float32")

    def call(self, td):
        td = self.mpnn(td)
        pooled = keras.ops.mean(td["nodes"], axis=0, keepdims=True)
        return self.head(pooled)


nodes = keras.ops.convert_to_tensor([[1.0, 0.0], [2.0, -1.0], [3.0, 1.0]], dtype="float32")
edges = keras.ops.convert_to_tensor([[0.5, -1.0], [1.5, 2.0], [-0.3, 0.7]], dtype="float32")
senders = keras.ops.convert_to_tensor([0, 1, 2], dtype="int32")
receivers = keras.ops.convert_to_tensor([1, 2, 0], dtype="int32")
td = {
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
y = keras.ops.convert_to_tensor([[1.0]], dtype="float32")

model = TinyMPNN()
_ = model(td)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    run_eagerly=True,
    jit_compile=False,
)
model.train_on_batch(td, y)
'''

    env = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
