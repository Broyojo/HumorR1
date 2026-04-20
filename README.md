# humor-rlhf

## Setup (PACE)

Run the setup script once after cloning:

```bash
bash scripts/setup.sh
source ~/.bashrc
```

This installs [uv](https://docs.astral.sh/uv/), creates scratch cache
directories, adds the required environment variables to your `~/.bashrc`,
and runs `uv sync` to install Python and project dependencies. It's safe
to re-run.

## Usage

Run scripts:
```bash
uv run python scripts/<script_name>.py
```

Add dependencies:
```bash
uv add <package_name>
```

## GRPO training (stub pipeline)

Smoke-test the GRPO loop on `google/gemma-4-E2B-it` with a random reward
and a tiny inline dataset. We'll promote this to the real New Yorker data +
Bradley-Terry reward model once those are ready.

Stack: plain HF transformers + PEFT/LoRA + TRL `GRPOTrainer` + vLLM in
**colocate** mode (rollouts on the same GPU as training). Unsloth is not
used — its `fast_inference=True` vLLM path does not yet support Gemma 4,
so it would force slow HF-generate rollouts. See the Unsloth Gemma 4 GRPO
reference notebook at [docs/grpo.py](docs/grpo.py) for the hyperparameter
baseline — [scripts/train_grpo.py](scripts/train_grpo.py) mirrors its knobs.

Interactive (1 GPU on whatever node you're on):
```bash
uv run python scripts/train_grpo.py
```

Via SLURM on PACE ICE:
```bash
sbatch scripts/train_grpo.slurm
```

Checkpoints land under `$CKPT_ROOT` (default `~/scratch/humor-rlhf/checkpoints`)
and the HF model cache under `$HF_HOME` (default `~/scratch/huggingface`, set
by `scripts/setup.sh`). `$HOME` on PACE ICE only has a 30 GB quota, so keep
all large artifacts on scratch.
