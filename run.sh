#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p secrets logs runs results .cache

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f secrets/hf_token ]]; then
    export HF_TOKEN="$(cat secrets/hf_token)"
  fi
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[fatal] HF_TOKEN is required (export it or create secrets/hf_token)" >&2
  exit 42
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "[fatal] NVIDIA driver not available" >&2
  exit 42
fi

export BASE_CACHE_DIR="${BASE_CACHE_DIR:-$PWD/.cache}"
mkdir -p "$BASE_CACHE_DIR"/{triton,torch,torch_extensions,inductor,hf,vllm,pip}
export XDG_CACHE_HOME="$BASE_CACHE_DIR"
export TRITON_CACHE_DIR="$BASE_CACHE_DIR/triton"
export TORCH_HOME="$BASE_CACHE_DIR/torch"
export TORCH_EXTENSIONS_DIR="$BASE_CACHE_DIR/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$BASE_CACHE_DIR/inductor"
export HF_HOME="$BASE_CACHE_DIR/hf"
export VLLM_CACHE_DIR="$BASE_CACHE_DIR/vllm"
export VLLM_USAGE_STATS=0
export VLLM_CACHE_ROOT="$BASE_CACHE_DIR/vllm"

mkdir -p "$PWD/bin"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PWD/bin" UV_NO_MODIFY_PATH=1 sh
  export PATH="$PWD/bin:$PATH"
  uv --version
fi

export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_AUTH_DIR="$PWD/share/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/bin/uvpython"
export PATH="$PWD/bin/uvpython:$PATH"

uv sync

if ! command -v singularity >/dev/null 2>&1 && ! command -v apptainer >/dev/null 2>&1; then
  echo "[fatal] singularity or apptainer is required" >&2
  exit 42
fi

export RUN_ID="${RUN_ID:-qwen3-${SLURM_JOB_ID:-manual}}"
export LITELLM_MODEL_REGISTRY_PATH="$ROOT/config/litellm_model_registry.json"

PORT="$(uv run python loop.py pick-port)"
export VLLM_PORT="$PORT"

echo "Using RUN_ID=$RUN_ID"
echo "Using VLLM_PORT=$VLLM_PORT"

override_cmd="${1:-full}"
uv run python loop.py "$override_cmd"
