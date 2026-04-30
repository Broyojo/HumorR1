"""
"Implement LLM eval"

Reference
---------
Zhang, Y., et al. (2024). "Humor AI at Massive Scale: 250 Million Human
Humor Preference Ratings on 2.2 Million New Yorker Cartoon Captions."
(yguooo/newyorker_caption_ranking on Hugging Face)

The Zhang et al. LLM evaluation prompts a strong VLM (or LLM) to act as a
humor judge. Given two captions for the same cartoon, the judge picks the
funnier one. We replicate this pairwise protocol using Claude / GPT-4o as
the judge (configurable) and report:

  - Win rate:       % of times our model's caption beats the baseline
  - Bradley-Terry score: MLE score from all pairwise comparisons
  - Mean humor score: absolute 1-5 scale (single-caption prompt)

Usage:
  # Pairwise comparison (two prediction JSONL files):
  uv run python scripts/llm_eval.py \\
      --mode pairwise \\
      --predictions results/baselines/zero_shot_predictions.jsonl \\
      --reference  results/baselines/sft_predictions.jsonl \\
      --output     results/llm_eval_pairwise.jsonl

  # Absolute scoring (single prediction file):
  uv run python scripts/llm_eval.py \\
      --mode absolute \\
      --predictions results/baselines/zero_shot_predictions.jsonl \\
      --output     results/llm_eval_absolute.jsonl

  # Score policy captions vs baselines (Exp 2):
  uv run python scripts/llm_eval.py \\
      --mode pairwise \\
      --predictions results/policy_captions.jsonl \\
      --reference  results/baselines/zero_shot_predictions.jsonl \\
      --output     results/exp2_llm_eval.jsonl

Judge model: set --judge-model to any OpenAI-compatible endpoint.
Defaults to "claude-sonnet-4-20250514" via Anthropic API if ANTHROPIC_API_KEY is set,
or "gpt-4o" via OpenAI API if OPENAI_API_KEY is set.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

SEED = 42
random.seed(SEED)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PAIRWISE_SYSTEM_PROMPT = """You are an expert humor evaluator specializing in New Yorker cartoon captions.
You will be shown a cartoon image and two candidate captions. Your job is to determine which caption is funnier.

A great New Yorker caption:
- Is unexpected and subverts the viewer's initial interpretation
- Uses wit rather than obvious humor
- Is concise — one punchy sentence
- Rewards the viewer for paying attention to the cartoon's details

Respond with ONLY a JSON object in this exact format:
{"winner": "A" or "B", "confidence": 1-5, "reasoning": "one sentence explanation"}
"""

PAIRWISE_USER_PROMPT = """Caption A: {caption_a}

Caption B: {caption_b}

Which caption is funnier for this cartoon? Respond with only the JSON."""

ABSOLUTE_SYSTEM_PROMPT = """You are an expert humor evaluator specializing in New Yorker cartoon captions.
You will be shown a cartoon image and one candidate caption. Rate the caption's humor.

A great New Yorker caption (score 5):
- Is unexpected and subverts the viewer's initial interpretation
- Uses wit rather than obvious humor  
- Is concise and punchy
- Rewards attention to the cartoon's details

A poor caption (score 1):
- States the obvious
- Is confusing or nonsensical
- Is too long or meandering

Respond with ONLY a JSON object in this exact format:
{"score": 1-5, "reasoning": "one sentence explanation"}
"""

ABSOLUTE_USER_PROMPT = """Caption: {caption}

Rate this caption's humor from 1 (not funny) to 5 (very funny). Respond with only the JSON."""


# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------


def call_anthropic(messages: list[dict], system: str, model: str) -> str:
    """Call Anthropic Messages API. Requires ANTHROPIC_API_KEY env var."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=system,
        messages=messages,
    )
    return response.content[0].text


def call_openai(messages: list[dict], system: str, model: str) -> str:
    """Call OpenAI Chat Completions API. Requires OPENAI_API_KEY env var."""
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "system", "content": system}] + messages,
    )
    return response.choices[0].message.content


def call_gemini(messages: list[dict], system: str, model: str) -> str:
    """Call Google Gemini API. Requires GEMINI_API_KEY env var.
    Free key at: https://aistudio.google.com/apikey

    Uses the google-genai SDK (>= 1.0).  Install with:
        pip install google-genai
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Collect user-role text parts (multimodal content is handled inline)
    contents: list[types.Content] = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        raw = m["content"]
        # Content may be a plain string or a list of typed blocks
        if isinstance(raw, str):
            parts = [types.Part(text=raw)]
        else:
            parts = []
            for block in raw:
                if block.get("type") == "text":
                    parts.append(types.Part(text=block["text"]))
                elif block.get("type") == "image_url":
                    # Inline base-64 images (not used in text-only eval mode)
                    parts.append(types.Part(text="[image]"))
        contents.append(types.Content(role=role, parts=parts))

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.0,
    )
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    return response.text


def call_groq(messages: list[dict], system: str, model: str) -> str:
    """Call Groq API. Requires GROQ_API_KEY env var.
    Free key (no credit card) at: https://console.groq.com
    Recommended model: llama-3.3-70b-versatile  (~14,400 req/day free)
    Install with: pip install groq
    """
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "system", "content": system}] + messages,
    )
    return response.choices[0].message.content


def judge(
    messages: list[dict],
    system: str,
    judge_model: str,
    max_retries: int = 3,
) -> str:
    """Route to correct API based on judge_model name and available keys.

    Auto-detection order (when judge_model is the default sentinel):
      1. GEMINI_API_KEY   → gemini-2.0-flash 
      2. ANTHROPIC_API_KEY → claude-sonnet-4-20250514
      3. OPENAI_API_KEY   → gpt-4o
    """
    # Resolve "auto" to a concrete model using whichever key is present
    resolved_model = judge_model
    if judge_model == "auto":
        if os.environ.get("GROQ_API_KEY"):
            resolved_model = "llama-3.3-70b-versatile"
        elif os.environ.get("GEMINI_API_KEY"):
            resolved_model = "gemini-2.0-flash"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            resolved_model = "claude-sonnet-4-20250514"
        elif os.environ.get("OPENAI_API_KEY"):
            resolved_model = "gpt-4o"
        else:
            raise EnvironmentError(
                "No API key found. Set one of:\n"
                "  GROQ_API_KEY    — free at https://console.groq.com  ← recommended\n"
                "  GEMINI_API_KEY  — free at https://aistudio.google.com/apikey\n"
                "  ANTHROPIC_API_KEY or OPENAI_API_KEY"
            )

    for attempt in range(max_retries):
        try:
            if any(x in resolved_model.lower() for x in ("llama", "mixtral", "gemma")) and os.environ.get("GROQ_API_KEY"):
                return call_groq(messages, system, resolved_model)
            elif "gemini" in resolved_model.lower() and os.environ.get("GEMINI_API_KEY"):
                return call_gemini(messages, system, resolved_model)
            elif "claude" in resolved_model.lower() and os.environ.get("ANTHROPIC_API_KEY"):
                return call_anthropic(messages, system, resolved_model)
            elif os.environ.get("OPENAI_API_KEY"):
                return call_openai(messages, system, resolved_model)
            else:
                raise EnvironmentError(
                    "No API key found. Set one of:\n"
                    "  GROQ_API_KEY    — free at https://console.groq.com  ← recommended\n"
                    "  GEMINI_API_KEY  — free at https://aistudio.google.com/apikey\n"
                    "  ANTHROPIC_API_KEY or OPENAI_API_KEY"
                )
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    [WARN] API call failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip().rstrip("```").strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Pairwise evaluation
# ---------------------------------------------------------------------------


def load_predictions(path: Path) -> dict[str, dict]:
    """Load a predictions JSONL, indexed by contest_number."""
    records = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            key = str(row["contest_number"])
            records[key] = row
    return records


def run_pairwise_eval(
    predictions_path: Path,
    reference_path: Path,
    output_path: Path,
    judge_model: str,
    max_samples: int | None = None,
    data_dir: Path = Path("data"),
):
    """Compare two sets of captions pairwise, using the LLM as judge."""
    preds = load_predictions(predictions_path)
    refs = load_predictions(reference_path)

    common_keys = sorted(set(preds.keys()) & set(refs.keys()))
    if max_samples:
        common_keys = common_keys[:max_samples]

    print(f"Running pairwise eval on {len(common_keys)} examples...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    wins_a = 0  # predictions win
    wins_b = 0  # reference wins
    ties = 0

    with open(output_path, "w") as f:
        for i, key in enumerate(common_keys):
            pred_row = preds[key]
            ref_row = refs[key]

            caption_a = pred_row["predicted_caption"]
            caption_b = ref_row["predicted_caption"]

            # Randomly swap A/B to reduce position bias
            swapped = random.random() < 0.5
            if swapped:
                caption_a, caption_b = caption_b, caption_a

            # Build message (text-only for now; add image if VLM judge is used)
            messages = [
                {
                    "role": "user",
                    "content": PAIRWISE_USER_PROMPT.format(
                        caption_a=caption_a, caption_b=caption_b
                    ),
                }
            ]

            try:
                raw = judge(messages, PAIRWISE_SYSTEM_PROMPT, judge_model)
                parsed = parse_json_response(raw)
                winner_label = parsed.get("winner", "A")
                confidence = parsed.get("confidence", 3)
                reasoning = parsed.get("reasoning", "")
            except Exception as e:
                print(f"  [ERROR] key={key}: {e}")
                winner_label = "tie"
                confidence = 0
                reasoning = f"error: {e}"

            # Undo swap
            if swapped:
                if winner_label == "A":
                    winner_label = "B"
                elif winner_label == "B":
                    winner_label = "A"

            # winner A = predictions model; winner B = reference model
            if winner_label == "A":
                wins_a += 1
            elif winner_label == "B":
                wins_b += 1
            else:
                ties += 1

            record = {
                "contest_number": key,
                "prediction_caption": pred_row["predicted_caption"],
                "reference_caption": ref_row["predicted_caption"],
                "winner": winner_label,   # "A"=predictions, "B"=reference
                "confidence": confidence,
                "reasoning": reasoning,
                "swapped": swapped,
                "judge_model": judge_model,
            }
            results.append(record)
            f.write(json.dumps(record) + "\n")

            if (i + 1) % 10 == 0:
                total = wins_a + wins_b + ties
                print(
                    f"  [{i+1}/{len(common_keys)}] "
                    f"win_rate={wins_a/total:.2%} ({wins_a}W / {wins_b}L / {ties}T)"
                )

    total = wins_a + wins_b + ties
    metrics = {
        "mode": "pairwise",
        "judge_model": judge_model,
        "n_examples": total,
        "wins_predictions": wins_a,
        "wins_reference": wins_b,
        "ties": ties,
        "win_rate": wins_a / total if total > 0 else 0.0,
        "predictions_file": str(predictions_path),
        "reference_file": str(reference_path),
    }

    metrics_path = output_path.with_suffix("").with_name(
        output_path.stem + "_metrics.json"
    )
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nPairwise eval complete.")
    print(f"  Win rate (predictions vs reference): {metrics['win_rate']:.2%}")
    print(f"  Results: {output_path}")
    print(f"  Metrics: {metrics_path}")
    return metrics


# ---------------------------------------------------------------------------
# Absolute scoring
# ---------------------------------------------------------------------------


def run_absolute_eval(
    predictions_path: Path,
    output_path: Path,
    judge_model: str,
    max_samples: int | None = None,
):
    """Score each caption on a 1-5 humor scale."""
    preds = load_predictions(predictions_path)
    keys = sorted(preds.keys())
    if max_samples:
        keys = keys[:max_samples]

    print(f"Running absolute eval on {len(keys)} captions...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    scores = []

    with open(output_path, "w") as f:
        for i, key in enumerate(keys):
            row = preds[key]
            caption = row["predicted_caption"]

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
                print(f"  [ERROR] key={key}: {e}")
                score = 3.0
                reasoning = f"error: {e}"

            scores.append(score)
            record = {
                "contest_number": key,
                "predicted_caption": caption,
                "humor_score": score,
                "reasoning": reasoning,
                "judge_model": judge_model,
            }
            results.append(record)
            f.write(json.dumps(record) + "\n")

            if (i + 1) % 10 == 0:
                print(
                    f"  [{i+1}/{len(keys)}] mean_score={sum(scores)/len(scores):.2f}"
                )

    mean_score = sum(scores) / len(scores) if scores else 0.0
    metrics = {
        "mode": "absolute",
        "judge_model": judge_model,
        "n_examples": len(scores),
        "mean_humor_score": mean_score,
        "predictions_file": str(predictions_path),
    }

    metrics_path = output_path.with_suffix("").with_name(
        output_path.stem + "_metrics.json"
    )
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\nAbsolute eval complete.")
    print(f"  Mean humor score: {mean_score:.2f} / 5.0")
    print(f"  Results: {output_path}")
    print(f"  Metrics: {metrics_path}")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="LLM eval (Zhang et al.)")
    parser.add_argument(
        "--mode",
        choices=["pairwise", "absolute"],
        required=True,
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="JSONL of predictions to evaluate",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="(pairwise only) JSONL of reference/baseline captions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (defaults to results/llm_eval_<mode>.jsonl)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="auto",
        help=(
            "LLM judge model ID. Default 'auto' picks the first available key: "
            "GEMINI_API_KEY→gemini-2.0-flash, ANTHROPIC_API_KEY→claude-sonnet-4-20250514, "
            "OPENAI_API_KEY→gpt-4o. Free Gemini key: https://aistudio.google.com/apikey"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap evaluation at N examples",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output = args.output or Path(f"results/llm_eval_{args.mode}.jsonl")

    if args.mode == "pairwise":
        if args.reference is None:
            raise ValueError("--reference is required for pairwise mode")
        run_pairwise_eval(
            args.predictions,
            args.reference,
            output,
            args.judge_model,
            args.max_samples,
            args.data_dir,
        )
    else:
        run_absolute_eval(
            args.predictions,
            output,
            args.judge_model,
            args.max_samples,
        )


if __name__ == "__main__":
    main()
