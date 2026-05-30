#!/usr/bin/env python3
"""Install DGL by auto-detecting the best wheel index for this machine.

This script inspects:
- OS / architecture
- installed PyTorch version
- CUDA runtime version (if available)

It then probes known DGL wheel index URLs and installs from the first matching
index using `uv pip install`.
"""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
import sys
import urllib.request


def _torch_info():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is not importable. Install torch before installing DGL."
        ) from exc

    torch_version = torch.__version__.split("+")[0]
    m = re.match(r"(\d+)\.(\d+)", torch_version)
    if not m:
        raise RuntimeError(f"Cannot parse torch version: {torch.__version__}")
    torch_mm = f"{m.group(1)}.{m.group(2)}"

    cuda_version = getattr(torch.version, "cuda", None)
    cuda_tag = None
    if cuda_version:
        cm = re.match(r"(\d+)\.(\d+)", cuda_version)
        if cm:
            cuda_tag = f"cu{cm.group(1)}{cm.group(2)}"

    return torch.__version__, torch_mm, cuda_version, cuda_tag


def _url_exists(url: str, timeout: int = 8) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def _torch_version_fallbacks(torch_mm: str) -> list[str]:
    major_s, minor_s = torch_mm.split(".")
    major = int(major_s)
    minor = int(minor_s)

    candidates = [f"{major}.{m}" for m in range(minor, -1, -1)]
    common_supported = ["2.8", "2.7", "2.6", "2.5", "2.4", "2.3", "2.2", "2.1", "2.0"]
    for v in common_supported:
        if v not in candidates:
            candidates.append(v)
    return candidates


def _candidate_urls(torch_mm: str, cuda_tag: str | None) -> list[str]:
    urls = []
    for tv in _torch_version_fallbacks(torch_mm):
        base = f"https://data.dgl.ai/wheels/torch-{tv}"
        if cuda_tag:
            urls.append(f"{base}/{cuda_tag}/repo.html")
        urls.append(f"{base}/repo.html")
        urls.append(f"{base}/cpu/repo.html")
    return urls


def _run(cmd: list[str], dry_run: bool) -> int:
    print("+", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-install DGL for current platform")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    sys_name = platform.system().lower()
    machine = platform.machine().lower()

    if sys_name != "linux":
        print(f"Info: detected non-Linux platform: {sys_name}")
    if machine not in {"x86_64", "amd64", "aarch64", "arm64"}:
        print(f"Warning: uncommon machine architecture: {machine}")

    torch_full, torch_mm, cuda_version, cuda_tag = _torch_info()
    print(f"Detected torch: {torch_full}")
    print(f"Detected torch major.minor: {torch_mm}")
    print(f"Detected CUDA runtime: {cuda_version or 'none'}")

    urls = _candidate_urls(torch_mm, cuda_tag)
    selected = None
    for url in urls:
        print(f"Probing: {url}")
        if _url_exists(url):
            selected = url
            break

    if selected is None:
        print("Could not find a matching DGL wheel index URL.")
        print("Tried:")
        for url in urls[:12]:
            print(f"  - {url}")
        if len(urls) > 12:
            print(f"  ... and {len(urls) - 12} more")
        print("You can still try: uv pip install dgl")
        return 2

    print(f"Using DGL wheel index: {selected}")
    code = _run(["uv", "pip", "install", "dgl", "-f", selected], dry_run=args.dry_run)
    if code != 0:
        print("Install from detected index failed; trying plain PyPI install...")
        code = _run(["uv", "pip", "install", "dgl"], dry_run=args.dry_run)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
