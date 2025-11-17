#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?model id}"
PORT="${2:?port}"
TP="${3:?tensor_parallel_size}"
LOGDIR="${4:-logs/vllm_${SLURM_JOB_ID:-manual}}"
mkdir -p "$LOGDIR"

# Require GPUs
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[fatal] nvidia-smi not found - GPU is required" >&2
  exit 1
fi
if ! nvidia-smi >/dev/null 2>&1; then
  echo "[fatal] nvidia-smi failed - GPU not usable" >&2
  exit 1
fi

# sanity check torch cuda inside the uv env
uv run python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("[fatal] CUDA is required for vLLM", file=sys.stderr)
    sys.exit(1)
PY

# Start vLLM OpenAI server using uv run
exec uv run python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port "$PORT" \
  --model "$MODEL" \
  --dtype auto \
  --quantization fp8 \
  --tensor-parallel-size "$TP" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --enforce-eager \
  --max-num-seqs 64 \
  --disable-log-requests \
  > "$LOGDIR/server.stdout.log" 2> "$LOGDIR/server.stderr.log"
