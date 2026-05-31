"""tf_gnns benchmark implementation for ogbg-molhiv."""

from __future__ import annotations

import tensorflow as tf
import keras

from tf_gnns.models.gcn import SparseGCNConv

from .common import now_ms, summarize_times


class TfGNNsGCN(keras.Model):
    def __init__(self, hidden_dim: int, layer_jit_compile: bool = False):
        super().__init__()
        self.gcn1 = SparseGCNConv(
            hidden_dim, activation="relu", jit_compile=layer_jit_compile
        )
        self.gcn2 = SparseGCNConv(
            hidden_dim, activation="relu", jit_compile=layer_jit_compile
        )
        self.head = keras.layers.Dense(1)

    def call(self, td, training=False):
        td = self.gcn1(td)
        td = self.gcn2(td)
        pooled = keras.ops.mean(td["nodes"], axis=0, keepdims=True)
        logits = self.head(pooled)
        return logits


def benchmark_tf_gnns(
    tf_samples,
    hidden_dim: int,
    lr: float,
    warmup_steps: int,
    bench_steps: int,
    layer_jit_compile: bool = False,
    train_step_jit_compile: bool = False,
    sample_mode: str = "single",
    use_tf_function: bool = False,
):
    model = TfGNNsGCN(hidden_dim=hidden_dim, layer_jit_compile=layer_jit_compile)
    opt = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

    def _train_step_impl(td):
        y = tf.reshape(td["labels"], (1, 1))
        with tf.GradientTape() as tape:
            logits = model(td, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    if use_tf_function:
        train_step = tf.function(
            _train_step_impl,
            jit_compile=train_step_jit_compile,
            reduce_retracing=True,
        )
    else:
        train_step = _train_step_impl

    n = len(tf_samples)

    def pick(i):
        if sample_mode == "single":
            return tf_samples[0]
        return tf_samples[i % n]

    for i in range(warmup_steps):
        _ = train_step(pick(i))

    step_times = []
    for i in range(bench_steps):
        t0 = now_ms()
        _ = train_step(pick(i))
        t1 = now_ms()
        step_times.append(t1 - t0)

    metrics = summarize_times(step_times)
    mode = "tf_eager"
    if use_tf_function:
        mode = "tf_function"
    if use_tf_function and train_step_jit_compile:
        mode = "tf_function_jit"
    if layer_jit_compile:
        mode = mode + "+layer_jit"
    device = "gpu" if tf.config.list_physical_devices("GPU") else "cpu"
    metrics.update({"framework": "tf_gnns", "mode": mode, "device": device})
    return metrics
