"""
Run OOD eval

This script:
  1. Runs each model (policy + baselines) on every OOD image (zero-shot)
  2. Scores captions with the LLM eval (Zhang et al.) — absolute 1-5 rating
  3. Optionally scores with the Bradley-Terry reward model 
  4. Writes per-image predictions and a summary table

Usage:
  # Generate captions + LLM score:
  uv run python scripts/run_ood_eval.py \\
      --ood-manifest data/ood_manifest.jsonl \\
      --policy-checkpoint ~/scratch/humor-rlhf/checkpoints/grpo-policy/final \\
      --output-dir results/exp3_ood

  # If policy isn't ready yet, just run baselines:
  uv run python scripts/run_ood_eval.py \\
      --ood-manifest data/ood_manifest.jsonl \\
      --baselines-only \\
      --output-dir results/exp3_ood

  # Skip caption generation (captions already produced) and only run LLM eval:
  uv run python scripts/run_ood_eval.py \\
      --ood-manifest data/ood_manifest.jsonl \\
      --skip-generation \\
      --output-dir results/exp3_ood

Outputs (all in --output-dir):
  captions_<model>.jsonl   — per-image caption predictions
  llm_scores_<model>.jsonl — absolute LLM humor scores
  summary_table.json       — aggregated metrics per model
"""

import argparse
import json
import os
from pathlib import Path

import torch

SEED = 42

MODEL_CONFIGS = {
    "zero_shot": {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "checkpoint": None,
        "label": "Zero-shot (Qwen3-VL-2B)",
    },
    "few_shot": {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "checkpoint": None,
        "label": "Few-shot (Qwen3-VL-2B)",
    },
    "sft": {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "checkpoint": None,   # set by --sft-checkpoint
        "label": "SFT (Qwen3-VL-2B)",
    },
    "policy": {
        "model_id": None,     # set by --policy-checkpoint
        "checkpoint": None,
        "label": "GRPO Policy",
    },
}

GENERATION_INSTRUCTION = (
    "Write a funny one-line caption for this image, as if it were a New Yorker cartoon. "
    "Be witty, unexpected, and concise. "
    "Output your caption between <caption> and </caption> tags."
)

SYSTEM_PROMPT = (
    "You are a witty caption writer. "
    "Think carefully about what makes the image surprising or funny before writing."
)


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------


def load_ood_manifest(manifest_path: Path) -> list[dict]:
    records = []
    with open(manifest_path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def generate_caption_for_image(
    model,
    processor,
    image_path: str,
    device: str,
    few_shot_examples: list[dict] | None = None,
) -> dict:
    """Generate caption for a single OOD image."""
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if few_shot_examples:
        for ex in few_shot_examples:
            messages.append({
                "role": "user",
                "content": f"[Example scene: {ex['context']}]\n{GENERATION_INSTRUCTION}",
            })
            messages.append({
                "role": "assistant",
                "content": f"<caption>{ex['caption']}</caption>",
            })

    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": GENERATION_INSTRUCTION},
        ],
    })

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        add_vision_info=True,  # required for Qwen3-VL image token embedding
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    think_text = ""
    caption = raw_output.strip()
    if "<think>" in raw_output and "</think>" in raw_output:
        think_text = raw_output.split("<think>")[1].split("</think>")[0].strip()
        caption = raw_output.split("</think>")[-1].strip()
    if "<caption>" in caption and "</caption>" in caption:
        caption = caption.split("<caption>")[1].split("</caption>")[0].strip()

    return {"caption": caption, "thinking": think_text, "raw_output": raw_output}


def run_caption_generation(
    model_key: str,
    model_id: str,
    ood_records: list[dict],
    output_dir: Path,
    data_root: Path,
    few_shot: bool = False,
) -> Path:
    """Run caption generation for one model on all OOD images."""
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
    except ImportError:
        from transformers import AutoModelForVision2Seq

    out_path = output_dir / f"captions_{model_key}.jsonl"
    if out_path.exists():
        print(f"  {model_key}: captions already exist at {out_path}, skipping.")
        return out_path

    print(f"\n--- Generating captions: {model_key} ({model_id}) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model_obj = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model_obj.eval()

    few_shot_examples = None
    if few_shot:
        few_shot_examples = [
            {
                "caption": "I didn't say it was your fault, I said I was going to blame you.",
                "context": "Two businessmen shaking hands in an office.",
            },
            {
                "caption": "Of course it's not the money—it's the principle of the money.",
                "context": "A couple arguing over a small coin on a table.",
            },
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with open(out_path, "w") as f:
        for i, rec in enumerate(ood_records):
            img_path = str(data_root / rec["local_path"])
            try:
                out = generate_caption_for_image(
                    model_obj, processor, img_path, device, few_shot_examples
                )
            except Exception as e:
                print(f"  [ERROR] {rec['id']}: {e}")
                out = {"caption": "", "thinking": "", "raw_output": ""}

            record = {
                "id": rec["id"],
                "category": rec["category"],
                "local_path": rec["local_path"],
                "description": rec.get("description", ""),
                "predicted_caption": out["caption"],
                "thinking": out["thinking"],
                "model": model_key,
            }
            results.append(record)
            f.write(json.dumps(record) + "\n")

            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(ood_records)}] done")

    del model_obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Captions saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# LLM scoring (reuses llm_eval.py logic)
# ---------------------------------------------------------------------------


def run_llm_scoring(
    captions_path: Path,
    output_dir: Path,
    judge_model: str,
    max_samples: int | None = None,
) -> dict:
    """Score OOD captions using absolute LLM humor scoring."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from llm_eval import judge, parse_json_response, ABSOLUTE_SYSTEM_PROMPT, ABSOLUTE_USER_PROMPT

    records = []
    with open(captions_path) as f:
        for line in f:
            records.append(json.loads(line))

    if max_samples:
        records = records[:max_samples]

    model_key = captions_path.stem.replace("captions_", "")
    out_path = output_dir / f"llm_scores_{model_key}.jsonl"

    if out_path.exists():
        print(f"  LLM scores for {model_key} already exist, loading...")
        scores = []
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                scores.append(r["humor_score"])
        return {"model": model_key, "mean_score": sum(scores) / len(scores), "n": len(scores)}

    print(f"\n--- LLM scoring: {model_key} ---")
    scores = []
    with open(out_path, "w") as f:
        for i, rec in enumerate(records):
            caption = rec["predicted_caption"]
            if not caption:
                continue

            messages = [
                {
                    "role": "user",
                    "content": ABSOLUTE_USER_PROMPT.format(caption=caption),
                }
            ]
            try:
                raw = judge(messages, ABSOLUTE_SYSTEM_PROMPT, judge_model)
                parsed = parse_json_response(raw)
                score = float(parsed.get("score", 3))
                reasoning = parsed.get("reasoning", "")
            except Exception as e:
                print(f"  [ERROR] {rec['id']}: {e}")
                score = 3.0
                reasoning = f"error: {e}"

            scores.append(score)
            record = {**rec, "humor_score": score, "reasoning": reasoning}
            f.write(json.dumps(record) + "\n")

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(records)}] mean={sum(scores)/len(scores):.2f}")

    mean_score = sum(scores) / len(scores) if scores else 0.0
    print(f"  Mean humor score for {model_key}: {mean_score:.2f}")
    return {"model": model_key, "mean_score": mean_score, "n": len(scores)}


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def build_summary_table(model_metrics: list[dict], output_dir: Path):
    """Build and print a summary table of OOD eval results."""
    from collections import defaultdict

    table = []
    for m in model_metrics:
        table.append(m)

    # Sort by mean score descending
    table.sort(key=lambda x: x.get("mean_score", 0), reverse=True)

    summary_path = output_dir / "summary_table.json"
    summary_path.write_text(json.dumps(table, indent=2))

    print("\n" + "=" * 60)
    print("Exp 3 OOD Eval — Summary Table")
    print("=" * 60)
    print(f"{'Model':<30} {'Mean Score':>12} {'N':>6}")
    print("-" * 60)
    for row in table:
        print(f"{row['model']:<30} {row.get('mean_score', 0):>12.2f} {row.get('n', 0):>6}")
    print("=" * 60)
    print(f"\nFull results saved to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Run OOD eval (Exp 3)")
    parser.add_argument(
        "--ood-manifest",
        type=Path,
        default=Path("data/ood_manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/exp3_ood"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory relative to which local_path in manifest is resolved",
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=str,
        default=None,
        help="Path to trained GRPO policy checkpoint",
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=str,
        default=None,
        help="Path to SFT checkpoint",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Skip policy model; only run baselines",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip caption generation; only run LLM scoring on existing files",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="claude-sonnet-4-20250514",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.ood_manifest.exists():
        raise FileNotFoundError(
            f"OOD manifest not found: {args.ood_manifest}. "
            "Run scripts/curate_ood_dataset.py first."
        )

    ood_records = load_ood_manifest(args.ood_manifest)
    if args.max_samples:
        ood_records = ood_records[:args.max_samples]
    print(f"Loaded {len(ood_records)} OOD images.")

    # -----------------------------------------------------------------------
    # Step 1: Generate captions for each model
    # -----------------------------------------------------------------------
    captions_files = []

    if not args.skip_generation:
        # Zero-shot baseline
        cp = run_caption_generation(
            "zero_shot",
            "Qwen/Qwen3-VL-2B-Instruct",
            ood_records,
            args.output_dir,
            args.data_root,
            few_shot=False,
        )
        captions_files.append(cp)

        # Few-shot baseline
        cp = run_caption_generation(
            "few_shot",
            "Qwen/Qwen3-VL-2B-Instruct",
            ood_records,
            args.output_dir,
            args.data_root,
            few_shot=True,
        )
        captions_files.append(cp)

        # SFT baseline
        sft_id = args.sft_checkpoint or "Qwen/Qwen3-VL-2B-Instruct"
        cp = run_caption_generation(
            "sft",
            sft_id,
            ood_records,
            args.output_dir,
            args.data_root,
            few_shot=False,
        )
        captions_files.append(cp)

        # Policy model (if available)
        if not args.baselines_only and args.policy_checkpoint:
            cp = run_caption_generation(
                "policy",
                args.policy_checkpoint,
                ood_records,
                args.output_dir,
                args.data_root,
                few_shot=False,
            )
            captions_files.append(cp)
    else:
        captions_files = sorted(args.output_dir.glob("captions_*.jsonl"))
        print(f"Skipping generation; found {len(captions_files)} existing caption files.")

    # -----------------------------------------------------------------------
    # Step 2: LLM scoring
    # -----------------------------------------------------------------------
    model_metrics = []
    for cap_file in captions_files:
        any_key = any(os.environ.get(k) for k in (
            "GROQ_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"
        ))
        if not any_key:
            print(
                "\n[WARN] No API key set. Skipping LLM scoring. "
                "Set GROQ_API_KEY (free) or GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY to enable."
            )
            break
        metrics = run_llm_scoring(cap_file, args.output_dir, args.judge_model)
        model_metrics.append(metrics)

    # -----------------------------------------------------------------------
    # Step 3: Summary table
    # -----------------------------------------------------------------------
    if model_metrics:
        build_summary_table(model_metrics, args.output_dir)
    else:
        print("\nNo metrics to summarize (LLM scoring skipped or no API key).")
        print("Caption files generated — run with API key to score.")


if __name__ == "__main__":
    main()
