#!/usr/bin/env bash
# One-shot project setup for humor-rlhf on PACE.
# Safe to re-run.

set -euo pipefail

BASHRC="$HOME/.bashrc"
HF_EXPORT='export HF_HOME="$HOME/scratch/huggingface"'

echo "==> Checking for PACE scratch directory"
if [ ! -d "$HOME/scratch" ]; then
  echo "ERROR: ~/scratch not found."
  echo "This script assumes you're on the PACE cluster, where ~/scratch is"
  echo "a symlink to your storage allocation. If you're not on PACE, ask"
  echo "the project owner how to adapt this setup."
  exit 1
fi

echo "==> Creating Hugging Face cache directory"
mkdir -p "$HOME/scratch/huggingface"

echo "==> Installing uv (if missing)"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
else
  echo "    uv already installed, skipping"
fi

echo "==> Ensuring HF_HOME export in ~/.bashrc"
touch "$BASHRC"
if grep -qE 'HF_HOME.*scratch/huggingface' "$BASHRC"; then
  echo "    already set: $HF_EXPORT"
else
  if [ -s "$BASHRC" ] && [ -n "$(tail -c1 "$BASHRC")" ]; then
    echo "" >> "$BASHRC"
  fi
  echo "" >> "$BASHRC"
  echo "$HF_EXPORT" >> "$BASHRC"
  echo "    added: $HF_EXPORT"
fi

echo "==> Syncing project dependencies (uv sync)"
export HF_HOME="$HOME/scratch/huggingface"
export PATH="$PATH:$HOME/.local/bin"
uv sync

echo
echo "Setup complete. If this was your first run, 'source ~/.bashrc' now."
