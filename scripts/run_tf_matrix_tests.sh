#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_tf_matrix_tests.sh 2.17 2.18 2.19 2.20 2.21
# If no versions are passed, defaults to the compatibility matrix below.

TF_VERSIONS=("$@")
if [ ${#TF_VERSIONS[@]} -eq 0 ]; then
  TF_VERSIONS=("2.17" "2.18" "2.19" "2.20" "2.21")
fi

tfp_for_tf() {
  local key
  key=$(echo "$1" | cut -d. -f1-2)
  case "$key" in
    2.17) echo "0.24" ;;
    2.18) echo "0.25" ;;
    2.19) echo "0.25" ;;
    2.20) echo "0.25" ;;
    2.21) echo "0.25" ;;
    *)
      echo "Unsupported TensorFlow version: $1" >&2
      return 1
      ;;
  esac
}

needs_numpy_1x() {
  local minor
  minor=$(echo "$1" | cut -d. -f2)
  if [ "$minor" -le 14 ]; then
    echo "true"
  else
    echo "false"
  fi
}

needs_tf_keras() {
  local minor
  minor=$(echo "$1" | cut -d. -f2)
  if [ "$minor" -ge 16 ]; then
    echo "true"
  else
    echo "false"
  fi
}

for tf_ver in "${TF_VERSIONS[@]}"; do
  echo "==== Testing TensorFlow ${tf_ver} ===="
  tfp_ver=$(tfp_for_tf "$tf_ver")
  venv_dir=".venv-tf${tf_ver//./}"

  rm -rf "$venv_dir"
  uv venv --python 3.11 "$venv_dir"

  uv pip install --python "$venv_dir/bin/python" -e . --no-deps
  uv pip install --python "$venv_dir/bin/python" pytest
  uv pip install --python "$venv_dir/bin/python" "tensorflow==${tf_ver}.*" "tensorflow_probability==${tfp_ver}.*"

  if [ "$(needs_numpy_1x "$tf_ver")" = "true" ]; then
    uv pip install --python "$venv_dir/bin/python" "numpy<2"
  fi

  if [ "$(needs_tf_keras "$tf_ver")" = "true" ]; then
    uv pip install --python "$venv_dir/bin/python" "tf_keras==${tf_ver}.*"
  fi

  "$venv_dir/bin/python" -m pytest -v
done
