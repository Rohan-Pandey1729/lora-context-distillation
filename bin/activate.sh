#!/usr/bin/env bash
set -euo pipefail

# HF token required
mkdir -p secrets
if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f secrets/hf_token ]]; then export HF_TOKEN="$(cat secrets/hf_token)"; fi
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[fatal] HF_TOKEN is required" >&2; exit 42
fi
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# GPU required
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "[fatal] NVIDIA driver not available" >&2; exit 42
fi

# keep all caches local to the repo to avoid filling $HOME quotas
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

# install uv locally if missing
mkdir -p "$PWD/bin"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PWD/bin" UV_NO_MODIFY_PATH=1 sh
  export PATH="$PWD/bin:$PATH"
  uv --version
fi

export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_AUTH_DIR="$PWD/share/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/bin/uvpython"
export PATH="$PWD/bin/uvpython:$PATH"

if ! command -v python3.10 >/dev/null 2>&1; then
  echo "[fatal] python3.10 is required in PATH" >&2; exit 42
fi

# venv and caches under project
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export UV_CACHE_DIR="$PWD/.cache/uv"
export PIP_CACHE_DIR="$PWD/.cache/pip"
mkdir -p .hf .cache logs runs results
export VLLM_CACHE_ROOT="$BASE_CACHE_DIR/vllm"

uv sync


# verify HF and CUDA in the env
uv run python - <<'PY'
import os, sys, torch
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).whoami()
if not torch.cuda.is_available():
    print("[fatal] CUDA is required but not available", file=sys.stderr); sys.exit(42)
print("CUDA OK - devices:", torch.cuda.device_count())
PY

# require apptainer or singularity - no aliasing
if ! command -v singularity >/dev/null 2>&1 && ! command -v apptainer >/dev/null 2>&1; then
  echo "[fatal] singularity or apptainer is required" >&2; exit 42
fi
