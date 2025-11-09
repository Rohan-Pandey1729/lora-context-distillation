#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:?model id}"
PORT="${2:?port}"
TP="${3:?tensor_parallel_size}"
LOGDIR="${4:-logs/vllm_${SLURM_JOB_ID:-manual}}"
mkdir -p "$LOGDIR"
exec python -m vllm.entrypoints.openai.api_server \
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
