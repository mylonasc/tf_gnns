import importlib.util
import os
import subprocess
import sys

import pytest


def _module_available(name):
    return importlib.util.find_spec(name) is not None


def _run(code):
    env = dict(os.environ)
    env["KERAS_BACKEND"] = "jax"
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _run_with_env(code, extra_env):
    env = dict(os.environ)
    env["KERAS_BACKEND"] = "jax"
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


_GRAPH_TD_WITH_GLOBAL = r'''
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
    "global_attr": keras.ops.zeros((1, 1), dtype="float32"),
    "global_reps_for_edges": keras.ops.convert_to_tensor([0, 0, 0], dtype="int32"),
    "global_reps_for_nodes": keras.ops.convert_to_tensor([0, 0, 0], dtype="int32"),
}
'''


@pytest.mark.skipif(not _module_available("jax"), reason="JAX is not installed")
def test_jax_compiled_dense_train_on_batch_succeeds():
    code = r'''
import keras

x = keras.ops.ones((4, 3), dtype="float32")
y = keras.ops.ones((4, 1), dtype="float32")

model = keras.Sequential([keras.layers.Dense(8, activation="relu"), keras.layers.Dense(1)])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.MeanSquaredError(),
    run_eagerly=False,
    jit_compile=True,
)
loss = model.train_on_batch(x, y)
print(float(keras.ops.convert_to_numpy(loss)))
'''
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr or proc.stdout


@pytest.mark.skipif(not _module_available("jax"), reason="JAX is not installed")
@pytest.mark.parametrize("with_global", ["0", "1"])
def test_jax_compiled_graphnet_mpnn_train_on_batch_succeeds(with_global):
    code = r'''
import os
import keras
from tf_gnns.models.graphnet import GraphNetMPNN_MLP


class TinyMPNN(keras.Model):
    def __init__(self):
        super().__init__()
        self.mpnn = GraphNetMPNN_MLP(units=8, core_units=8, core_steps=1)
        self.head = keras.layers.Dense(1, dtype="float32")

    def build(self, input_shape):
        self.mpnn.build(input_shape)
        self.head.build((None, self.mpnn.node_output_size))
        super().build(input_shape)

    def call(self, td, training=False):
        del training
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
    "global_reps_for_edges": keras.ops.convert_to_tensor([0, 0, 0], dtype="int32"),
    "global_reps_for_nodes": keras.ops.convert_to_tensor([0, 0, 0], dtype="int32"),
}
if os.environ["WITH_GLOBAL"] == "1":
    td["global_attr"] = keras.ops.zeros((1, 1), dtype="float32")
else:
    td["global_attr"] = None

y = keras.ops.convert_to_tensor([[1.0]], dtype="float32")

model = TinyMPNN()
input_shape = {k: (None if v is None else tuple(v.shape)) for k, v in td.items()}
model.build(input_shape)
_ = model(td)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    run_eagerly=False,
    jit_compile=True,
)
model.train_on_batch(td, y)
'''
    env = dict(os.environ)
    env["WITH_GLOBAL"] = with_global
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env={**env, "KERAS_BACKEND": "jax"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


@pytest.mark.skipif(not _module_available("jax"), reason="JAX is not installed")
def test_jax_compiled_graphnet_mlp_train_on_batch_fails_same_signature():
    code = r'''
import keras
from tf_gnns.models.graphnet import GraphNetMLP


class TinyGraphNet(keras.Model):
    def __init__(self):
        super().__init__()
        self.gn = GraphNetMLP(units=8, core_units=8, core_steps=1)
        self.head = keras.layers.Dense(1, dtype="float32")

    def build(self, input_shape):
        self.gn.build(input_shape)
        self.head.build((None, self.gn.node_output_size))
        super().build(input_shape)

    def call(self, td, training=False):
        del training
        td = self.gn(td)
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
    "global_attr": keras.ops.zeros((1, 1), dtype="float32"),
    "global_reps_for_edges": keras.ops.convert_to_tensor([0, 0, 0], dtype="int32"),
    "global_reps_for_nodes": keras.ops.convert_to_tensor([0, 0, 0], dtype="int32"),
}
y = keras.ops.convert_to_tensor([[1.0]], dtype="float32")

model = TinyGraphNet()
input_shape = {k: tuple(v.shape) for k, v in td.items()}
model.build(input_shape)
_ = model(td)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    run_eagerly=False,
    jit_compile=True,
)
model.train_on_batch(td, y)
'''
    proc = _run(code)
    assert proc.returncode != 0
    combined = f"{proc.stderr}\n{proc.stdout}"
    assert "Unable to automatically build the model" in combined
    assert "_error_repr" in combined


@pytest.mark.skipif(not _module_available("jax"), reason="JAX is not installed")
def test_jax_jitted_stateless_mpnn_step_succeeds():
    code = r'''
import keras
import jax
from tf_gnns.models.graphnet import GraphNetMPNN_MLP


class TinyMPNN(keras.Model):
    def __init__(self):
        super().__init__()
        self.mpnn = GraphNetMPNN_MLP(units=8, core_units=8, core_steps=1)
        self.head = keras.layers.Dense(1, dtype="float32")

    def build(self, input_shape):
        self.mpnn.build(input_shape)
        self.head.build((None, self.mpnn.node_output_size))
        super().build(input_shape)

    def call(self, td, training=False):
        del training
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
    "global_attr": keras.ops.zeros((1, 1), dtype="float32"),
    "global_reps_for_edges": keras.ops.convert_to_tensor([0, 0, 0], dtype="int32"),
    "global_reps_for_nodes": keras.ops.convert_to_tensor([0, 0, 0], dtype="int32"),
}
y = keras.ops.convert_to_tensor([[1.0]], dtype="float32")

model = TinyMPNN()
input_shape = {k: tuple(v.shape) for k, v in td.items()}
model.build(input_shape)
_ = model(td)

loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
opt = keras.optimizers.Adam(learning_rate=1e-3)
opt.build(model.trainable_variables)
tv = [v.value for v in model.trainable_variables]
ntv = [v.value for v in model.non_trainable_variables]
ov = [v.value for v in opt.variables]

def _step(tv, ntv, ov, td, y):
    def _loss_and_aux(tv, ntv):
        logits, new_ntv = model.stateless_call(tv, ntv, td, training=True)
        loss = loss_fn(y, logits)
        return loss, new_ntv

    (loss, new_ntv), grads = jax.value_and_grad(_loss_and_aux, has_aux=True)(tv, ntv)
    new_ov, new_tv = opt.stateless_apply(ov, grads, tv)
    return loss, new_tv, new_ntv, new_ov

jitted_step = jax.jit(_step)
loss, tv, ntv, ov = jitted_step(tv, ntv, ov, td, y)
loss_val = float(keras.ops.convert_to_numpy(loss))
if not (loss_val == loss_val):
    raise RuntimeError("loss is NaN")
print(loss_val)
'''
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr or proc.stdout


@pytest.mark.skipif(not _module_available("jax"), reason="JAX is not installed")
@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_jax_mpnn_benchmark_runner_executes_and_reports_json(mode):
    code = r'''
import json
import os
import runpy
import sys
import types
import numpy as np


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
    "--backend", "jax",
    "--mode", os.environ["BENCH_MODE"],
    "--steps", "1",
    "--warmup", "0",
    "--max-graphs", "1",
    "--feature-dtype", "float32",
    "--index-dtype", "int32",
]

runpy.run_path("benchmarks/ogbg_molhiv_mpnn/run_tfgnns_backend.py", run_name="__main__")
'''
    proc = _run_with_env(code, {"BENCH_MODE": mode})
    if mode == "eager":
        assert proc.returncode == 0, proc.stderr or proc.stdout
        payload = proc.stdout.strip().splitlines()[-1]
        parsed = __import__("json").loads(payload)
        assert parsed["backend"] == "jax"
        assert parsed["mode"] == "keras_eager"
    else:
        assert proc.returncode == 0, proc.stderr or proc.stdout
        payload = proc.stdout.strip().splitlines()[-1]
        parsed = __import__("json").loads(payload)
        assert parsed["backend"] == "jax"
        assert parsed["mode"] == "keras_compiled"


@pytest.mark.skipif(not _module_available("jax"), reason="JAX is not installed")
def test_jax_eval_shape_graphindep_succeeds_but_graphnet_core_message_passing_fails():
    code = (
        r'''import keras
import jax
from tf_gnns.models.graphnet import GraphIndep
from tf_gnns import GraphNet, make_mlp_graphnet_functions
'''
        + _GRAPH_TD_WITH_GLOBAL
        + r'''
graph_indep = GraphIndep(units_out=4)
_ = graph_indep(td)
jax.eval_shape(graph_indep, td)

gn_args = make_mlp_graphnet_functions(
    8,
    node_input_size=2,
    node_output_size=2,
    edge_input_size=2,
    edge_output_size=2,
    use_global_input=True,
    use_global_to_edge=True,
    use_global_to_node=True,
    create_global_function=True,
    global_input_size=1,
    global_output_size=1,
    graph_indep=False,
    aggregation_function="mean",
)
gn = GraphNet(**gn_args)
_ = gn.eval_tensor_dict(td)
jax.eval_shape(lambda x: gn.eval_tensor_dict(x), td)
'''
    )
    proc = _run(code)
    assert proc.returncode != 0
    combined = f"{proc.stderr}\n{proc.stdout}"
    assert "ConcretizationTypeError" in combined or "concrete value" in combined


@pytest.mark.skipif(not _module_available("jax"), reason="JAX is not installed")
def test_jax_eval_shape_graphnet_mpnn_layer_succeeds():
    code = (
        r'''import keras
import jax
from tf_gnns.models.graphnet import GraphNetMPNN_MLP
'''
        + _GRAPH_TD_WITH_GLOBAL
        + r'''
mpnn = GraphNetMPNN_MLP(units=8, core_units=8, core_steps=1)
_ = mpnn(td)
jax.eval_shape(mpnn, td)
'''
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr or proc.stdout
