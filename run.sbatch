#!/bin/bash
#SBATCH --job-name=qwen3-loop
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --gpus=l40s:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --chdir=.
#SBATCH --export=ALL

set -euo pipefail
umask 077

mkdir -p logs

unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
export NO_PROXY="localhost,127.0.0.1,::1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

# Run ID defaults to qwen3-$SLURM_JOB_ID in run.sh; override here if desired.
./run.sh
