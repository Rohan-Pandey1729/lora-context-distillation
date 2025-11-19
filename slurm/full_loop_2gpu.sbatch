#!/bin/bash
#SBATCH --job-name=qwen3-loop
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --gpus=l40s:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=2:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --chdir=.
#SBATCH --export=ALL

set -euo pipefail
umask 077
mkdir -p logs

# HF token required for the whole job
: "${HF_TOKEN:=$(cat secrets/hf_token 2>/dev/null || true)}"
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[fatal] HF_TOKEN is required. Export HF_TOKEN or create secrets/hf_token" >&2
  exit 42
fi
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# RUN_ID defaults to slurm job id
export RUN_ID="${RUN_ID:-qwen3-${SLURM_JOB_ID:-manual}}"

# bootstrap uv env and whatever checks you already do there
source bin/activate.sh

# make sure uv is on PATH
export PATH="$PWD/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[fatal] uv not found on PATH after bin/activate.sh" >&2
  exit 42
fi

UV="uv"

# Read config for model and port using uv run
PORT="$($UV run python - <<'PY'
from pipeline.util import load_conf
print(load_conf()["ports"]["vllm"])
PY
)"

MODEL_B="$($UV run python - <<'PY'
from pipeline.util import load_conf
print(load_conf()["model_b_base"])
PY
)"

# Ensure vLLM port is free
$UV run python bin/kill_port.py "$PORT" || true
LOGDIR="logs/vllm_${SLURM_JOB_ID}"

# Launch vLLM server - see note below about start_vllm_server.sh
bash bin/start_vllm_server.sh "$MODEL_B" "$PORT" 2 "$LOGDIR" &
VLLM_PID=$!

# Wait for readiness
$UV run python - <<PY
from pipeline.util import wait_vllm_ready
wait_vllm_ready(port=int("$PORT"), timeout_s=900)
print("vLLM ready")
PY

# SWE stage - produces preds.json and all-preds.jsonl and uploads continuously
$UV run python -m pipeline.swe_runner

# Strip <think> and produce SFT JSONL for A
$UV run python -m pipeline.strip_thinking \
  --preds_json "runs/${RUN_ID}/swe/preds.json" \
  --out_jsonl  "runs/${RUN_ID}/sft/sft_qwenA_from_B_mini.jsonl"

# Train A with Unsloth and push checkpoints and merged snapshots
$UV run python -m pipeline.train_unsloth_lora

# Apply linear DIFF to produce new B and push it
$UV run python -m pipeline.apply_diff_linear

$UV run python - <<'PY'
from pipeline.util import load_conf
from pipeline.hf_sync import upload_path, ensure_repo
import os
cfg = load_conf()
run_id = cfg["run_id"]
repo = cfg["repos_fmt"]["b_new_model"]
ensure_repo(repo, "model")
upload_path(repo, f"runs/{run_id}/B_new", "model")
print("Uploaded new B")
PY

# Stop vLLM
kill -9 "$VLLM_PID" 2>/dev/null || true
$UV run python bin/kill_port.py "$PORT" || true

# Snapshot logs
$UV run python -m pipeline.snap_and_sync logs || true

echo "[done] full loop completed"
