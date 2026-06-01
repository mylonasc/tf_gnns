"""TensorFlow runtime compatibility helpers.

Utilities in this package are intentionally lightweight and side-effect free
unless called explicitly.
"""

from .runtime_cuda_path import maybe_patch_ld_library_path_for_tensorflow

__all__ = ["maybe_patch_ld_library_path_for_tensorflow"]
