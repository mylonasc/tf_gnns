"""tf_gnns GraphNetMPNN benchmark implementation for ogbg-molhiv."""

from __future__ import annotations

import keras
import tensorflow as tf

from tf_gnns.models.graphnet import GraphNetMPNN_MLP

from .common import now_ms, summarize_times


class TfGNNsMPNN(keras.Model):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mpnn = GraphNetMPNN_MLP(units=hidden_dim, core_units=hidden_dim, core_steps=2)
        self.head = keras.layers.Dense(1)

    def call(self, td, training=False):
        td = self.mpnn(td)
        pooled = keras.ops.mean(td["nodes"], axis=0, keepdims=True)
        return self.head(pooled)


def benchmark_tf_gnns(
    tf_samples,
    hidden_dim: int,
    lr: float,
    warmup_steps: int,
    bench_steps: int,
    sample_mode: str = "single",
    use_tf_function: bool = False,
):
    model = TfGNNsMPNN(hidden_dim=hidden_dim)
    opt = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

    def _train_step_impl(td, y):
        y = tf.reshape(y, (1, 1))
        with tf.GradientTape() as tape:
            logits = model(td, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    train_step = _train_step_impl
    if use_tf_function:
        train_step = tf.function(_train_step_impl, jit_compile=False, reduce_retracing=True)

    n = len(tf_samples)

    def pick(i):
        if sample_mode == "single":
            return tf_samples[0]
        return tf_samples[i % n]

    for i in range(warmup_steps):
        td, y = pick(i)
        _ = train_step(td, y)

    step_times = []
    for i in range(bench_steps):
        td, y = pick(i)
        t0 = now_ms()
        _ = train_step(td, y)
        t1 = now_ms()
        step_times.append(t1 - t0)

    metrics = summarize_times(step_times)
    mode = "keras_eager" if not use_tf_function else "keras_compiled"
    device = "gpu" if tf.config.list_physical_devices("GPU") else "cpu"
    metrics.update(
        {
            "framework": "tf_gnns",
            "mode": mode,
            "device": device,
            "param_dtype": str(model.head.kernel.dtype),
        }
    )
    return metrics
