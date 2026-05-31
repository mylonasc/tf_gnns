"""Thin backend-ops facade over Keras 3 ops.

Most functions are direct wrappers around ``keras.ops`` so core library code can
avoid binding directly to TensorFlow symbols.

Current fallback matrix:
- ``segment_sum``: native ``keras.ops``
- ``segment_max``: native ``keras.ops``
- ``segment_mean``: composed from ``segment_sum`` + ``bincount``
- ``segment_min``: TensorFlow fallback only (until ``keras.ops.segment_min`` exists)

This module should stay intentionally small and declarative.
"""

import keras
import warnings


def active_backend():
    """Return the active Keras backend name."""
    return keras.backend.backend()


def has_native_segment_min():
    """Return whether the current Keras backend exposes ``segment_min``."""
    return hasattr(keras.ops, "segment_min")


_SEGMENT_MIN_EMULATION_WARNED = False


def _warn_segment_min_emulation():
    global _SEGMENT_MIN_EMULATION_WARNED
    if not _SEGMENT_MIN_EMULATION_WARNED:
        warnings.warn(
            "Using emulated segment_min via -segment_max(-x). "
            "This may be slower or numerically different than a native backend op.",
            RuntimeWarning,
            stacklevel=2,
        )
        _SEGMENT_MIN_EMULATION_WARNED = True


def _as_int_tensor(x):
    return keras.ops.cast(keras.ops.convert_to_tensor(x), "int32")


def gather(params, indices, axis=0):
    return keras.ops.take(params, _as_int_tensor(indices), axis=axis)


def concat(xs, axis=-1):
    return keras.ops.concatenate(xs, axis=axis)


def first_dim(x):
    return keras.ops.shape(x)[0]


def zeros_like(x):
    return keras.ops.zeros_like(x)


def zeros(shape, dtype="float32"):
    return keras.ops.zeros(shape, dtype=dtype)


def stack(xs, axis=0):
    return keras.ops.stack(xs, axis=axis)


def squeeze(x, axis=None):
    return keras.ops.squeeze(x, axis=axis)


def expand_dims(x, axis=0):
    return keras.ops.expand_dims(x, axis=axis)


def convert_to_tensor(x, dtype=None):
    return keras.ops.convert_to_tensor(x, dtype=dtype)


def identity(x):
    return x + 0


def shape(x):
    return keras.ops.shape(x)


def is_tensor(x):
    return keras.ops.is_tensor(x)


def reduce_all(x, axis=None):
    return keras.ops.all(x, axis=axis)


def reduce_mean(x, axis=0):
    return keras.ops.mean(x, axis=axis)


def reduce_sum(x, axis=0):
    return keras.ops.sum(x, axis=axis)


def reduce_max(x, axis=0):
    return keras.ops.max(x, axis=axis)


def reduce_min(x, axis=0):
    return keras.ops.min(x, axis=axis)


def segment_sum(values, indices, num_groups):
    return keras.ops.segment_sum(values, _as_int_tensor(indices), num_groups)


def segment_max(values, indices, num_groups):
    return keras.ops.segment_max(values, _as_int_tensor(indices), num_groups)


def _present_segment_mask(indices, num_groups, dtype):
    indices = _as_int_tensor(indices)
    present = keras.ops.segment_max(keras.ops.ones_like(indices), indices, num_groups)
    present = keras.ops.clip(present, 0, 1)
    present = keras.ops.cast(present, dtype)
    return keras.ops.expand_dims(present, axis=-1)


def _zero_absent_segments(reduced, indices, num_groups):
    present = _present_segment_mask(indices, num_groups, reduced.dtype)
    zeros = keras.ops.zeros_like(reduced)
    return keras.ops.where(present > 0, reduced, zeros)


def segment_max_or_zero(values, indices, num_groups):
    reduced = segment_max(values, indices, num_groups)
    return _zero_absent_segments(reduced, indices, num_groups)


def segment_mean(values, indices, num_groups):
    indices = _as_int_tensor(indices)
    sums = segment_sum(values, indices, num_groups)
    one_col = keras.ops.ones((keras.ops.shape(indices)[0], 1), dtype=sums.dtype)
    counts = segment_sum(one_col, indices, num_groups)
    counts = keras.ops.maximum(counts, keras.ops.ones_like(counts))
    return sums / counts


def segment_min(values, indices, num_groups):
    indices = _as_int_tensor(indices)
    if has_native_segment_min():
        return keras.ops.segment_min(values, indices, num_groups)
    if active_backend() == "tensorflow":
        import tensorflow as tf

        return tf.math.unsorted_segment_min(values, indices, num_groups)
    _warn_segment_min_emulation()
    return -segment_max(-values, indices, num_groups)


def segment_min_or_zero(values, indices, num_groups):
    reduced = segment_min(values, indices, num_groups)
    return _zero_absent_segments(reduced, indices, num_groups)
