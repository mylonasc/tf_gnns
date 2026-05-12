#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
cd "$ROOT_DIR"

uv sync --locked --group dev
uv run pytest -v --cov=tf_gnns --cov-report=term-missing --cov-report=xml:coverage.xml
uv run python scripts/update_coverage_badge.py coverage.xml doc/shields/coverage.json
