# Backends

Read this when choosing TensorFlow, Torch, or JAX Keras backends.

## Rules

- `tf_gnns` uses Keras 3 and a small `tf_gnns.backend_ops` facade for backend-portable tensor operations.
- Choose a Keras backend before importing Keras-heavy modules.
- TensorFlow is the broadest tested backend for the full package and examples.
- Torch and JAX paths are intended for backend portability smoke tests and compatible Keras operations.
- TensorFlow graph compilation should be applied at the application level with `tf.function`, not by changing `tf_gnns` internals.

## Backend Check Example

```python
from tf_gnns import backend_ops

print(backend_ops.active_backend())
```

## TensorFlow Compile Example

```python
import tensorflow as tf
from tf_gnns.models.graphnet import GraphNetMPNN_MLP

model = GraphNetMPNN_MLP(units=8, core_steps=1)

@tf.function
def forward(td):
    return model(td)
```
