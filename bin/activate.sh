#!/usr/bin/env bash
set -euo pipefail

# All caches live under the project working dir
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PWD/.mamba}"
export CONDA_PKGS_DIRS="$MAMBA_ROOT_PREFIX/pkgs"
export MAMBA_NO_RC=1
mkdir -p "$MAMBA_ROOT_PREFIX" "$PWD/bin"

# Ensure a real micromamba binary named "micromamba" - no .micromamba, no symlink
if [[ -e "$PWD/bin/.micromamba" || -L "$PWD/bin/micromamba" ]]; then
  rm -f "$PWD/bin/.micromamba" "$PWD/bin/micromamba"
fi
if [[ ! -x "$PWD/bin/micromamba" ]]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C "$PWD" bin/micromamba
  chmod +x "$PWD/bin/micromamba"
fi

# Initialize shell hook, then use the micromamba shell function for env ops
eval "$("$PWD/bin/micromamba" shell hook -s bash)"

# Create or update the environment - fail fast on solver errors
micromamba create -y -f env/environment.yml
micromamba activate qwen3-loop

# Local caches and HF acceleration
mkdir -p .hf .cache logs runs results
export HF_HOME="$PWD/.hf"
export TRANSFORMERS_CACHE="$PWD/.hf/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=1

# HF token is required - no fallback to anonymous
if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f secrets/hf_token ]]; then
    export HF_TOKEN="$(cat secrets/hf_token)"
  fi
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[fatal] HF_TOKEN is required. Export HF_TOKEN or put it in secrets/hf_token" >&2
  exit 42
fi
# Some libs read this name
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# Optional: map apptainer to singularity name if only apptainer is installed
if ! command -v singularity >/dev/null 2>&1 && command -v apptainer >/dev/null 2>&1; then
  ln -sf "$(command -v apptainer)" "$PWD/bin/singularity"
  export PATH="$PWD/bin:$PATH"
fi

# Verify HF login
python - <<'PY'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
who = api.whoami()
print(f"HF login ok as {who.get('name') or who.get('email')}")
PY

# Require CUDA to be usable - do not allow CPU fallback
python - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print(f"[fatal] PyTorch import failed: {e}", file=sys.stderr); sys.exit(1)
if not torch.cuda.is_available():
    print("[fatal] CUDA is required but torch.cuda.is_available() is false", file=sys.stderr)
    sys.exit(1)
print(f"CUDA visible devices: {torch.cuda.device_count()}")
PY
