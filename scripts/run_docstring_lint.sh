#!/usr/bin/env bash

set -euo pipefail

uv sync --locked --group dev
uvx ruff check tf_gnns --select D || true
