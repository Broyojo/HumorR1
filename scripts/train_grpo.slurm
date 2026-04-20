#!/usr/bin/env bash
# SLURM submission wrapper for scripts/train_grpo.py on PACE ICE.
#
# Submit with:
#     sbatch scripts/train_grpo.slurm
#
# PACE ICE ice-gpu partition, account=coc, qos=coc-ice (see ~/.claude memory
# or `pace-quota` / `sinfo -s` to reconfirm).

#SBATCH --job-name=gemma4-grpo-humor
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --partition=ice-gpu
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:V100:1
# ice-gpu MaxTime is 16:00:00. GPU options (per `sinfo`): v100, rtx_6000,
# a40, a100, mi210, l40s (48G), h100 (80G), h200 (141G). Lowercase names
# also accepted. Note: V100 is Volta → no bfloat16; the train script
# auto-switches to fp16 when DTYPE=float16 is set below.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Scratch-backed caches so we don't blow the 30 GB $HOME quota.
export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/scratch/triton_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR"

export PATH="$HOME/.local/bin:$PATH"

# V100 requires fp16 (no bfloat16 on Volta). Unset or override when
# switching back to Ampere+ GPUs.
export DTYPE="${DTYPE:-float16}"

nvidia-smi || true
echo "==> Launching GRPO training"
uv run python scripts/train_grpo.py "$@"
