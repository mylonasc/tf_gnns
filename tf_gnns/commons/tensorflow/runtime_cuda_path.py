"""Helpers to patch TensorFlow CUDA library lookup at runtime.

The main function in this module is designed for environments where:
- the system has CUDA 13 runtime/driver tooling, and
- the installed TensorFlow wheel is built against CUDA 12.x (or earlier), and
- CUDA/cuDNN shared libraries are installed via pip under
  ``site-packages/nvidia/*/lib``.

In that setup TensorFlow may fail to load GPU libraries unless those pip paths
are present in ``LD_LIBRARY_PATH``. ``maybe_patch_ld_library_path_for_tensorflow``
detects that condition and prepends the discovered library directories.
"""

from __future__ import annotations

import glob
import os
import re
import site
import subprocess
from typing import Optional


_PATCH_ALREADY_RUN = False
_PATCH_MARKER_ENV = "TFGNNS_TF_LD_LIBRARY_PATCHED"


def _major_from_version_string(version: str) -> Optional[int]:
    m = re.search(r"(\d+)\.(\d+)", version)
    if not m:
        return None
    return int(m.group(1))


def _detect_system_cuda_major() -> Optional[int]:
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
        )
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", text)
        if m:
            return _major_from_version_string(m.group(1))
    except Exception:
        pass

    try:
        proc = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = re.search(r"release\s+([0-9]+\.[0-9]+)", text)
        if m:
            return _major_from_version_string(m.group(1))
    except Exception:
        pass

    return None


def _detect_tf_build_cuda_major() -> Optional[int]:
    try:
        import tensorflow as tf

        build_info = tf.sysconfig.get_build_info()
        cuda_v = str(build_info.get("cuda_version", ""))
        return _major_from_version_string(cuda_v)
    except Exception:
        return None


def _discover_pip_nvidia_lib_paths() -> list[str]:
    lib_paths: list[str] = []
    for base in site.getsitepackages():
        matches = sorted(glob.glob(os.path.join(base, "nvidia", "*", "lib")))
        lib_paths.extend(matches)
    seen = set()
    deduped = []
    for p in lib_paths:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def maybe_patch_ld_library_path_for_tensorflow() -> bool:
    """Patch ``LD_LIBRARY_PATH`` for TensorFlow CUDA library discovery.

    This function is idempotent and applies the patch at most once per process.
    It also marks the environment with ``TFGNNS_TF_LD_LIBRARY_PATCHED=1`` so
    repeated calls in the same shell context can skip rework.

    Behavior:
    1. Detect whether the host appears to have CUDA 13 tooling.
    2. Detect TensorFlow wheel build CUDA major version.
    3. If host CUDA is 13+ and TensorFlow is built for CUDA <=12,
       prepend pip-installed NVIDIA lib directories to ``LD_LIBRARY_PATH``.

    Returns:
        ``True`` if ``LD_LIBRARY_PATH`` was modified in this call, else ``False``.
    """

    global _PATCH_ALREADY_RUN
    if _PATCH_ALREADY_RUN or os.environ.get(_PATCH_MARKER_ENV) == "1":
        return False

    _PATCH_ALREADY_RUN = True

    system_cuda_major = _detect_system_cuda_major()
    tf_cuda_major = _detect_tf_build_cuda_major()

    if system_cuda_major is None or tf_cuda_major is None:
        return False
    if system_cuda_major < 13:
        return False
    if tf_cuda_major > 12:
        return False

    lib_paths = _discover_pip_nvidia_lib_paths()
    if not lib_paths:
        return False

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_items = [p for p in current.split(":") if p]
    merged = lib_paths + [p for p in current_items if p not in lib_paths]
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)
    os.environ[_PATCH_MARKER_ENV] = "1"
    return True
