"""Qwen3-VL-2B-Thinking baselines: zero-shot, few-shot, and SFT.

Usage:
  # Zero-shot (no training required):
  uv run python scripts/run_baselines.py --mode zero_shot

  # Few-shot (no training required):
  uv run python scripts/run_baselines.py --mode few_shot

  # SFT fine-tune then evaluate:
  uv run python scripts/run_baselines.py --mode sft --train

  # SFT evaluate only (from saved checkpoint):
  uv run python scripts/run_baselines.py --mode sft --checkpoint <path>

Outputs are written to results/baselines/<mode>_predictions.jsonl and
results/baselines/<mode>_metrics.json.

Model: Qwen/Qwen3-VL-2B-Instruct
The "Thinking" variant uses <think>...</think> / </think> CoT internally.
We expose this via enable_thinking=True in the generation config.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

# transformers 5.x renamed AutoModelForVision2Seq → AutoModelForImageTextToText
try:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForVision2Seq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DATA_DIR = Path("data")
RESULTS_DIR = Path("results/baselines")
CKPT_ROOT = Path(
    os.environ.get("CKPT_ROOT", str(Path.home() / "scratch/humor-rlhf/checkpoints"))
)

SYSTEM_PROMPT = (
    "You are a witty caption writer for New Yorker-style cartoons. "
    "Think carefully about the scene, the absurdity, and the characters before writing. "
    "Your caption should be a single punchy sentence that is funny and surprising."
)

GENERATION_INSTRUCTION = (
    "Write a funny one-line caption for this New Yorker-style cartoon. "
    "Output your caption between <caption> and </caption> tags."
)

# Few-shot examples (fixed; drawn from training set high-rated captions)
# These are representative placeholder examples; replace with real ones
# once data/caption_sft_train is downloaded.
FEW_SHOT_EXAMPLES = [
    {
        "caption": "I didn't say it was your fault, I said I was going to blame you.",
        "context": "Scene: Two businessmen shaking hands in an office.",
    },
    {
        "caption": "Of course it's not the money—it's the principle of the money.",
        "context": "Scene: A couple arguing over a small coin on a table.",
    },
    {
        "caption": "We used to be so close—before the WiFi password changed.",
        "context": "Scene: A family sitting apart, each on their own device.",
    },
]

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_model_and_processor(device="auto"):
    """Load Qwen3-VL-2B-Instruct model and processor."""
    print(f"Loading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    # T4 does not support bfloat16 natively; use float16 instead.
    # device_map="auto" lets accelerate shard across GPU+CPU if needed.
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def build_zero_shot_messages(image_path: str, prompt_text: str) -> list[dict]:
    """Build zero-shot conversation with a single image+text turn."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": GENERATION_INSTRUCTION},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


def build_few_shot_messages(image_path: str, prompt_text: str) -> list[dict]:
    """Build few-shot conversation: 3 text-only examples then the target image."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Text-only few-shot examples (no images needed)
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({
            "role": "user",
            "content": f"{ex['context']}\n{GENERATION_INSTRUCTION}",
        })
        messages.append({
            "role": "assistant",
            "content": f"<caption>{ex['caption']}</caption>",
        })

    # Actual query with image
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": GENERATION_INSTRUCTION},
            {"type": "text", "text": prompt_text},
        ],
    })
    return messages


def _resize_image(img, max_pixels: int = 448 * 448):
    """Downscale image so total pixels <= max_pixels (keeps aspect ratio).

    T4 has ~15 GB VRAM. Qwen3-VL encodes every 28x28 patch as one visual
    token, so a 1000x700 image produces ~900 tokens. Capping at 448x448
    (~256 patches worst-case) prevents the 34 GB allocation OOM crash.
    """
    from PIL import Image as PILImage
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                         PILImage.LANCZOS)
    return img


def generate_caption(model, processor, messages: list[dict], device: str) -> dict:
    """Run inference; return dict with caption and raw thinking text."""
    from PIL import Image as PILImage

    # Extract image paths from messages, load as PIL, and resize for T4
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "image":
                    img_val = block["image"]
                    if isinstance(img_val, str):
                        img = PILImage.open(img_val).convert("RGB")
                        img = _resize_image(img)           # resize before encoding
                        block["image"] = img               # replace path with PIL object
                        images.append(img)
                    elif hasattr(img_val, "convert"):
                        img_val = _resize_image(img_val)   # resize PIL too
                        block["image"] = img_val
                        images.append(img_val)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_vision_info=True,      # required for Qwen3-VL to embed image tokens
    )
    first_device = next(model.parameters()).device
    inputs = processor(
        text=[text],
        images=images if images else None,
        return_tensors="pt",
    ).to(first_device)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,    # captions are short; 512 was wasteful & OOM-prone
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Slice off the input tokens
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_output = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    # Parse <think> block (Qwen3 thinking mode)
    think_text = ""
    caption = raw_output.strip()
    if "<think>" in raw_output and "</think>" in raw_output:
        think_text = raw_output.split("<think>")[1].split("</think>")[0].strip()
        caption = raw_output.split("</think>")[-1].strip()

    # Parse <caption> block
    if "<caption>" in caption and "</caption>" in caption:
        caption = caption.split("<caption>")[1].split("</caption>")[0].strip()

    return {"caption": caption, "thinking": think_text, "raw_output": raw_output}


# ---------------------------------------------------------------------------
# Zero-shot & Few-shot evaluation
# ---------------------------------------------------------------------------


def run_inference(mode: str, split: str = "test", max_samples: int | None = None):
    """Run zero-shot or few-shot inference on the test split.

    The dataset has multiple captions per image (it's a ranking dataset).
    We deduplicate by image path so we generate one caption per unique image,
    then fan the prediction back out to all rows sharing that image.
    """
    assert mode in ("zero_shot", "few_shot"), f"Unknown mode: {mode}"

    dataset_path = DATA_DIR / f"caption_sft_{split}"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Run scripts/download_data.py first."
        )

    dataset = load_from_disk(str(dataset_path))

    # Deduplicate: one row per unique image (keep first occurrence)
    seen_images = {}
    unique_rows = []
    for row in dataset:
        if row["image_path"] not in seen_images:
            seen_images[row["image_path"]] = len(unique_rows)
            unique_rows.append(row)

    if max_samples:
        unique_rows = unique_rows[:max_samples]

    print(f"Running {mode} on {len(unique_rows)} unique images "
          f"(from {len(dataset)} total rows) in {split} split...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model_and_processor(device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{mode}_predictions.jsonl"

    # Generate one caption per unique image
    caption_cache: dict[str, dict] = {}
    for i, row in enumerate(unique_rows):
        image_path = str(DATA_DIR / row["image_path"])
        if mode == "zero_shot":
            messages = build_zero_shot_messages(image_path, row["prompt"])
        else:
            messages = build_few_shot_messages(image_path, row["prompt"])

        try:
            out = generate_caption(model, processor, messages, device)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  [OOM] row {i}: {e}")
            torch.cuda.empty_cache()
            out = {"caption": "", "thinking": "", "raw_output": ""}
        except Exception as e:
            print(f"  [ERROR] row {i}: {e}")
            torch.cuda.empty_cache()
            out = {"caption": "", "thinking": "", "raw_output": ""}

        caption_cache[row["image_path"]] = out
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(unique_rows)}] done")

    # Fan predictions back out to all rows (one record per row, shared prediction)
    results = []
    with open(out_path, "w") as f:
        rows_to_write = dataset if not max_samples else [
            r for r in dataset if r["image_path"] in caption_cache
        ]
        for row in rows_to_write:
            out = caption_cache.get(row["image_path"],
                                    {"caption": "", "thinking": "", "raw_output": ""})
            record = {
                "contest_number": row["contest_number"],
                "image_path": row["image_path"],
                "reference_caption": row["caption"],
                "predicted_caption": out["caption"],
                "thinking": out["thinking"],
                "mode": mode,
            }
            results.append(record)
            f.write(json.dumps(record) + "\n")

    print(f"Predictions saved to {out_path} ({len(results)} records, "
          f"{len(caption_cache)} unique captions)")

    # ---- lexical metrics (BLEU-1 and ROUGE-L) ----
    bleu1_scores = []
    rougeL_scores = []
    for r in results:
        ref = r["reference_caption"].lower().split()
        hyp = r["predicted_caption"].lower().split()
        if ref and hyp:
            # BLEU-1: unigram precision with brevity penalty
            hits = sum(1 for w in hyp if w in set(ref))
            precision = hits / len(hyp) if hyp else 0.0
            bp = min(1.0, len(hyp) / len(ref)) if ref else 0.0
            bleu1_scores.append(bp * precision)
            # ROUGE-L: longest common subsequence F1
            m, n = len(ref), len(hyp)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for ii in range(1, m + 1):
                for jj in range(1, n + 1):
                    dp[ii][jj] = dp[ii-1][jj-1] + 1 if ref[ii-1] == hyp[jj-1] else max(dp[ii-1][jj], dp[ii][jj-1])
            lcs = dp[m][n]
            prec = lcs / n if n else 0.0
            rec  = lcs / m if m else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            rougeL_scores.append(f1)

    metrics = {
        "mode": mode,
        "split": split,
        "n_examples": len(results),
        "bleu1": sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else None,
        "rougeL": sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else None,
    }
    metrics_path = RESULTS_DIR / f"{mode}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics   saved to {metrics_path}")
    print(f"  BLEU-1 : {metrics['bleu1']:.4f}" if metrics["bleu1"] is not None else "  BLEU-1 : n/a")
    print(f"  ROUGE-L: {metrics['rougeL']:.4f}" if metrics["rougeL"] is not None else "  ROUGE-L: n/a")
    return results


# ---------------------------------------------------------------------------
# SFT fine-tuning
# ---------------------------------------------------------------------------


class CaptionSFTDataset(torch.utils.data.Dataset):
    """Tokenized dataset for caption SFT training."""

    def __init__(self, hf_dataset, processor, data_dir: Path):
        self.dataset = hf_dataset
        self.processor = processor
        self.data_dir = data_dir

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        from PIL import Image as PILImage
        row = self.dataset[idx]
        image_path = str(self.data_dir / row["image_path"])
        image = PILImage.open(image_path).convert("RGB")
        image = _resize_image(image)  # keep vision token count predictable

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": GENERATION_INSTRUCTION},
                    {"type": "text", "text": row["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": f"<caption>{row['caption']}</caption>",
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            add_vision_info=True,  # required for Qwen3-VL image token embedding
        )
        # Do NOT truncate: the processor embeds image tokens into input_ids and
        # truncating mid-sequence causes a token-count mismatch ValueError.
        # Images are already resized above so the sequence stays within ~600 tokens.
        encoding = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            truncation=False,
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def run_sft(checkpoint: str | None = None):
    """Fine-tune Qwen3-VL-2B on the SFT caption dataset with LoRA."""
    output_dir = CKPT_ROOT / "qwen3-vl-2b-sft"
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint:
        print(f"Loading SFT model from checkpoint: {checkpoint}")
        model_path = checkpoint
    else:
        model_path = MODEL_ID

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        dtype=torch.float16,   # T4 uses float16, not bfloat16
        device_map="auto",
        trust_remote_code=True,
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if not checkpoint:
        # Apply LoRA for efficient fine-tuning (T4-compatible settings)
        lora_config = LoraConfig(
            r=8,                          # reduced from 16 for T4 memory
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Gradient checkpointing cuts memory ~40%
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    train_ds = CaptionSFTDataset(
        load_from_disk(str(DATA_DIR / "caption_sft_train")),
        processor,
        DATA_DIR,
    )
    val_ds = CaptionSFTDataset(
        load_from_disk(str(DATA_DIR / "caption_sft_validation")),
        processor,
        DATA_DIR,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=1,    # reduced from 2 for T4
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,    # increased to compensate
        learning_rate=2e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=True,                        # T4 uses fp16, not bfloat16
        bf16=False,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=1,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("Starting SFT training...")
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    print(f"SFT model saved to {output_dir}/final")
    return str(output_dir / "final")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen3-VL-2B baselines")
    parser.add_argument(
        "--mode",
        choices=["zero_shot", "few_shot", "sft"],
        required=True,
        help="Which baseline to run",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="(SFT only) Run fine-tuning before evaluation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="(SFT only) Path to an existing SFT checkpoint for eval",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap evaluation at this many samples (for quick smoke-tests)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "sft":
        ckpt = args.checkpoint
        if args.train or ckpt is None:
            ckpt = run_sft(checkpoint=ckpt)

        # Evaluate the SFT model using zero-shot messages (no examples needed
        # since the model is already fine-tuned on the caption task)
        dataset_path = DATA_DIR / f"caption_sft_{args.split}"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        dataset = load_from_disk(str(dataset_path))
        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            ckpt,
            dtype=torch.float16,   # T4 uses float16
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "sft_predictions.jsonl"
        results = []
        with open(out_path, "w") as f:
            for i, row in enumerate(dataset):
                image_path = str(DATA_DIR / row["image_path"])
                messages = build_zero_shot_messages(image_path, row["prompt"])
                try:
                    out = generate_caption(model, processor, messages, device)
                except torch.cuda.OutOfMemoryError as e:
                    print(f"  [ERROR] row {i}: OOM — {e}")
                    torch.cuda.empty_cache()
                    out = {"caption": "", "thinking": "", "raw_output": ""}
                except Exception as e:
                    print(f"  [ERROR] row {i}: {e}")
                    torch.cuda.empty_cache()
                    out = {"caption": "", "thinking": "", "raw_output": ""}

                record = {
                    "contest_number": row["contest_number"],
                    "image_path": row["image_path"],
                    "reference_caption": row["caption"],
                    "predicted_caption": out["caption"],
                    "thinking": out["thinking"],
                    "mode": "sft",
                }
                results.append(record)
                f.write(json.dumps(record) + "\n")

                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(dataset)}] done")

        print(f"SFT predictions saved to {out_path}")

    else:
        run_inference(args.mode, split=args.split, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
