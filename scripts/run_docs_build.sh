#!/usr/bin/env bash

set -euo pipefail

uv sync --locked --group docs
uv run --group docs sphinx-build -b html docs docs/_build/html
