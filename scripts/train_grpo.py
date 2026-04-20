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
MAX_SEQ_LENGTH = 4096
LORA_RANK = 32

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
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
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
        # Optimizer & schedule (mirrors the Gemma 4 E2B notebook in docs/grpo.py).
        learning_rate=5e-5,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        # GRPO-specific.
        temperature=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=4,
        max_completion_length=MAX_SEQ_LENGTH - 512,
        max_steps=20,
        save_steps=20,
        logging_steps=1,
        epsilon=0.2,
        epsilon_high=0.28,
        delta=1.5,
        loss_type="bnpo",
        mask_truncated_completions=True,
        # Fast rollouts: vLLM runs colocated on the same GPU as the trainer.
        use_vllm=USE_VLLM,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35,
        report_to="none",
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
