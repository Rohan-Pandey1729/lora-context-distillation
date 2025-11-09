#!/usr/bin/env bash
set -euo pipefail
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.mambaforge}"
if ! command -v micromamba >/dev/null 2>&1; then
  mkdir -p bin
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
  mv bin/micromamba bin/.micromamba && ln -s .micromamba bin/micromamba
fi
eval "$(./bin/micromamba shell hook -s bash)"
./bin/micromamba create -y -f env/environment.yml || true
./bin/micromamba activate qwen3-loop
mkdir -p .hf .cache logs runs results
export HF_HOME="$PWD/.hf"
export TRANSFORMERS_CACHE="$PWD/.hf/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=1
if [ -f secrets/hf_token ]; then export HF_TOKEN="$(cat secrets/hf_token)"; fi
if ! command -v singularity >/dev/null 2>&1 && command -v apptainer >/dev/null 2>&1; then
  ln -sf "$(command -v apptainer)" bin/singularity
  export PATH="$PWD/bin:$PATH"
fi
python - <<'PY'
import os
from huggingface_hub import HfApi
tok=os.environ.get("HF_TOKEN","")
print("HF token present" if tok else "HF token missing - write secrets/hf_token")
if tok:
  HfApi(token=tok).whoami()
  print("HF login ok")
PY
