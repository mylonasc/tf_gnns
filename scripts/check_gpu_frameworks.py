#!/usr/bin/env python3
"""Verify TensorFlow and PyTorch are installed with GPU support."""

from __future__ import annotations

import json
import subprocess
import sys


def _run_probe(code: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def check_torch() -> bool:
    code = """
import json
import torch
payload = {
  'version': torch.__version__,
  'cuda_available': bool(torch.cuda.is_available()),
  'device_count': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
  'devices': [],
}
if payload['cuda_available']:
    for i in range(payload['device_count']):
        payload['devices'].append({
            'index': i,
            'name': torch.cuda.get_device_name(i),
            'capability': '.'.join(map(str, torch.cuda.get_device_capability(i))),
        })
print(json.dumps(payload))
"""
    rc, out, err = _run_probe(code)
    if rc != 0:
        print("[torch] import/probe failed")
        if err.strip():
            print(err.strip())
        return False

    payload = json.loads(out.strip())
    print(f"[torch] version: {payload['version']}")
    print(f"[torch] cuda available: {payload['cuda_available']}")
    if payload["cuda_available"]:
        print(f"[torch] gpu count: {payload['device_count']}")
        for d in payload["devices"]:
            print(
                f"[torch] gpu[{d['index']}]: {d['name']} capability={d['capability']}"
            )
    return bool(payload["cuda_available"])


def check_tensorflow() -> bool:
    code = """
import json
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
payload = {
  'version': tf.__version__,
  'gpu_available': len(gpus) > 0,
  'gpus': [g.name for g in gpus],
}
print(json.dumps(payload))
"""
    rc, out, err = _run_probe(code)
    if rc != 0:
        print("[tensorflow] import/probe failed")
        if err.strip():
            print(err.strip())
        return False

    payload = json.loads(out.strip())
    print(f"[tensorflow] version: {payload['version']}")
    print(f"[tensorflow] gpu available: {payload['gpu_available']}")
    for i, name in enumerate(payload["gpus"]):
        print(f"[tensorflow] gpu[{i}]: {name}")
    if err.strip():
        print("[tensorflow] probe stderr:")
        print(err.strip())
    return bool(payload["gpu_available"])


def main() -> int:
    print("Checking GPU support for torch and tensorflow...\n")
    torch_ok = check_torch()
    print("")
    tf_ok = check_tensorflow()
    print("")

    if torch_ok and tf_ok:
        print("PASS: torch and tensorflow both have GPU support.")
        return 0

    if not torch_ok and not tf_ok:
        print("FAIL: neither torch nor tensorflow reports GPU support.")
    elif not torch_ok:
        print("FAIL: torch does not report GPU support.")
    else:
        print("FAIL: tensorflow does not report GPU support.")
        print(
            "Hint: try running via scripts/run_with_tf_gpu to expose pip-installed CUDA/cuDNN libs."
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
