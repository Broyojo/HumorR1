"""GRPO training of Gemma 4 E2B-it for humor captioning via TRL + PEFT + vLLM.

This is the pipeline David owns (docs/tasks.md: "Implement GRPO pipeline
(stub reward)"). It runs against a tiny text-only stub dataset with a
random reward so the full RL loop can be validated before Yosie's New
Yorker reward model lands. Swap dataset + reward import when the real RM
is ready.

Design notes:
  - No Unsloth. Unsloth's `fast_inference=True` vLLM path doesn't support
    Gemma 4 yet (unsloth.ai/docs/models/gemma-4/train), so Unsloth buys us
    only training-step optimizations while forcing slow HF-generate
    rollouts. For GRPO, rollouts usually dominate wall-clock, so plain
    TRL + PEFT + vLLM colocate is faster end-to-end on 1 GPU.
  - Model is passed as a string; TRL auto-detects the arch via
    `config.architectures[0]` (Gemma4ForConditionalGeneration) in
    trl/trainer/utils.py::create_model_from_path.
  - LoRA applied via `peft_config=LoraConfig(...)`; GRPOTrainer handles
    adapter creation and reference-adapter duplication.
  - vLLM colocate shares the GPU with the trainer; tune
    `vllm_gpu_memory_utilization` (~0.35 on H100 80GB). Server mode would
    be faster but needs a 2nd GPU we don't have.
"""

import os
import sys
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).parent))
from reward_stub import compute_score


MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-4-E2B-it")
MAX_SEQ_LENGTH = 1024
# Rank choice per thinkingmachines.ai/blog/lora: RL absorbs ~1 bit/episode,
# so even rank 1 matches FullFT. Pick 32 as a safety margin without
# materially raising memory. Alpha stays fixed at 32 (not 2*rank) so the
# PEFT scale = alpha/r becomes rank-invariant per the same post.
LORA_RANK = 32
LORA_ALPHA = 32

# TRL 0.29.1 officially supports vLLM 0.10–0.12, but Gemma 4 needs vLLM
# >= 0.18. We install the newer vLLM and hope the API surface TRL uses is
# stable. If this breaks at runtime, flip USE_VLLM=0 to fall back to plain
# HF generate (slower but guaranteed).
USE_VLLM = os.environ.get("USE_VLLM", "1") == "1"

CKPT_ROOT = Path(
    os.environ.get("CKPT_ROOT", str(Path.home() / "scratch/humor-rlhf/checkpoints"))
)
OUTPUT_DIR = CKPT_ROOT / "gemma4-e2b-grpo-stub"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PROMPT = (
    "Write a funny one-line caption for a New Yorker style cartoon. "
    "Put your final caption between <caption></caption> tags."
)


def build_dataset(n_rows: int = 200) -> Dataset:
    """Tiny text-only stub. When real New Yorker data is ready, replace
    with a Dataset that has an `image` column and per-row cartoon context."""
    return Dataset.from_list(
        [{"prompt": [{"role": "user", "content": PROMPT}], "answer": 0}] * n_rows
    )


def reward_stub(completions, **kwargs):
    """TRL GRPO reward signature. Delegates to compute_score so swapping
    to the real reward model is a one-line change."""
    return [
        compute_score(
            data_source="humor-rlhf/stub",
            solution_str=c[0]["content"],
            ground_truth=None,
        )
        for c in completions
    ]


def main():
    # LoRA settings follow thinkingmachines.ai/blog/lora:
    #   - target_modules="all-linear": attention-only LoRA underperforms
    #     even at matched params; MLP/MoE layers must be adapted too.
    #     Bonus: this matches the inner nn.Linear inside Gemma 4's
    #     Gemma4ClippableLinear wrapper automatically.
    #   - lora_alpha=32 fixed (not scaled with r) so the alpha/r scale
    #     is rank-independent.
    #   - PEFT default init (A ~ Kaiming, B = 0) matches the post's
    #     recommendation.
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Volta (V100) lacks bfloat16; Ampere+ (A100/H100/L40S/H200) supports it.
    # Toggle via env: DTYPE=float16 for V100, DTYPE=bfloat16 elsewhere.
    dtype = os.environ.get("DTYPE", "bfloat16")
    use_bf16 = dtype == "bfloat16"

    training_args = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        # Model loading: TRL reads config.architectures[0] to pick the class.
        model_init_kwargs={"dtype": dtype},
        # Enable Gemma 4's reasoning mode (off by default) so the policy
        # can plan a format-correct caption before emitting tags.
        chat_template_kwargs={"enable_thinking": True},
        # Optimizer & schedule. LR ~10x a typical FullFT RL LR (~5e-6),
        # per thinkingmachines.ai/blog/lora — LoRA tolerates and benefits
        # from a higher LR than full fine-tuning.
        learning_rate=5e-5,
        weight_decay=0.001,
        lr_scheduler_type="constant",
        optim="adamw_8bit",
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        # GRPO-specific.
        temperature=1.0,
        # Keep the effective batch modest: the LoRA post warns that LoRA
        # pays a larger loss penalty than FullFT as batch size grows, and
        # the penalty does not shrink with higher rank.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=256,
        max_steps=200,
        save_steps=200,
        logging_steps=1,
        # KL penalty against the frozen reference adapter: keeps the
        # policy from drifting onto degenerate low-reward modes (seen
        # previously: dropped the literal "caption" entirely).
        beta=0.04,
        epsilon=0.2,
        epsilon_high=0.28,
        delta=1.5,
        loss_type="cispo",
        mask_truncated_completions=True,
        # Fast rollouts: vLLM runs colocated on the same GPU as the trainer.
        use_vllm=USE_VLLM,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.45,
        # Gemma 4's native context is 131k; vLLM tries to reserve KV
        # cache for the full length and OOMs on 24GB. Cap to what we
        # actually use (prompt + completion + slack).
        vllm_max_model_length=1024,
        report_to="wandb",
        run_name=os.environ.get("WANDB_RUN_NAME", "gemma4-e2b-grpo-stub"),
        logging_first_step=True,
        # Rollout visibility: print a few completions per log step and
        # push the full table to wandb so we can actually eyeball what
        # the policy is generating.
        log_completions=True,
        num_completions_to_print=2,
    )

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[reward_stub],
        args=training_args,
        train_dataset=build_dataset(),
        peft_config=lora_config,
    )
    trainer.train()

    lora_dir = OUTPUT_DIR / "lora_final"
    trainer.model.save_pretrained(str(lora_dir))
    if trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(str(lora_dir))
    print(f"Saved LoRA adapters to {lora_dir}")


if __name__ == "__main__":
    main()
