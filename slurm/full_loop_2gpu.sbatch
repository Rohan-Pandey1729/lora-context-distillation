#!/bin/bash
#SBATCH --job-name=qwen3-loop
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --gpus=l40s:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --chdir=.
#SBATCH --export=ALL

set -euo pipefail

eval "$(.micromamba shell hook --shell bash)"
source bin/activate.sh
: "${RUN_ID:?Set RUN_ID env var, e.g. RUN_ID=mini-$(date +%Y%m%d_%H%M%S)}"

export HF_TOKEN="$(cat secrets/hf_token)"
export PORT=$(python - <<PY
import socket
s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()
PY
)
echo "[info] RUN_ID=$RUN_ID PORT=$PORT"

# snapshot the exact code used for this run to HF
python -m pipeline.snap_and_sync code

# 1) start vLLM for model B on 2 GPUs
./bin/start_vllm_server.sh "Qwen/Qwen3-30B-A3B-Thinking-2507" "$PORT" 2 "logs/vllm_${SLURM_JOB_ID}" &
VLLM_PID=$!

# readiness check
python - <<PY
from pipeline.util import wait_vllm_ready
import os
wait_vllm_ready(int(os.environ["PORT"]))
print("vLLM is ready")
PY

# point mini-swe-agent at the local endpoint
python - <<PY
import os, yaml
with open("conf/mini_qwen_thinking.yaml","r") as f:
    c=yaml.safe_load(f)
c["model"]["model_kwargs"]["api_base"]=f"http://127.0.0.1:{os.environ[PORT]}/v1"
open("conf/mini_qwen_thinking.yaml","w").write(yaml.safe_dump(c))
PY

# 2) run SWEBench mini with continuous HF sync of preds.json, progress.json, all-preds.jsonl
python -m pipeline.swe_runner

# stop server and snapshot logs
python bin/kill_port.py "$PORT" || true
kill -9 $VLLM_PID || true
sleep 5
python -m pipeline.snap_and_sync logs

# 3) strip thinking and upload SFT jsonl
python - <<PY
from pipeline.util import load_conf
from pipeline.hf_sync import ensure_repo, upload_path
import subprocess, os
cfg=load_conf()
run_id=cfg["run_id"]
preds=f"runs/{run_id}/swe/preds.json"
out_jsonl=f"runs/{run_id}/sft/sft_qwenA_from_B_mini.jsonl"
os.makedirs(f"runs/{run_id}/sft", exist_ok=True)
subprocess.check_call(["python","-m","pipeline.strip_thinking","--preds_json",preds,"--out_jsonl",out_jsonl])
repo=cfg["repos_fmt"]["sft_dataset"]; ensure_repo(repo,"dataset"); upload_path(repo, out_jsonl, "dataset")
PY

# 4) train A on 2 GPUs + push ckpts and merged snapshots as we go
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
torchrun --nproc_per_node=2 -m pipeline.train_unsloth_lora

# snapshot logs after training
python -m pipeline.snap_and_sync logs

# 5) apply linear DIFF to get new B and upload
python -m pipeline.apply_diff_linear
python - <<PY
from pipeline.util import load_conf
from pipeline.hf_sync import upload_path, ensure_repo
import os
cfg=load_conf()
repo=cfg["repos_fmt"]["b_new_model"]
ensure_repo(repo,"model")
upload_path(repo, f"runs/{cfg[run_id]}/B_new", "model")
PY

# final log snapshot
python -m pipeline.snap_and_sync logs
echo "[done] full loop completed"
