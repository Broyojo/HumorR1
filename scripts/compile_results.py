"""Compile all experiment results into a unified table for the final report.

Reads outputs from:
  - results/baselines/           (zero_shot, few_shot, sft predictions + LLM scores)
  - results/exp2_human_eval.json (human eval, Exp 2)
  - results/exp3_ood/            (OOD eval, Exp 3)
  - results/llm_eval_*.json      (LLM eval on in-distribution test set)

Outputs:
  - results/tables/table_main.json      — machine-readable master table
  - results/tables/table_main.tex       — LaTeX table for the writeup
  - results/tables/table_exp1.tex       — Exp 1: RM agreement
  - results/tables/table_exp2.tex       — Exp 2: Human eval win rates
  - results/tables/table_exp3.tex       — Exp 3: OOD eval scores
  - results/tables/summary.md           — Markdown summary of all results

Usage:
  uv run python scripts/compile_results.py
  uv run python scripts/compile_results.py --results-dir results/ --output-dir results/tables/
"""

import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def safe_load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return None


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def load_llm_metrics(results_dir: Path) -> dict:
    """Load LLM eval metrics for baselines (in-distribution test set).

    llm_eval.py writes metrics to:
      results/llm_eval_absolute_<mode>_metrics.json   (absolute mode)
      results/llm_eval_pairwise_<mode>_metrics.json   (pairwise mode)

    Lexical metrics (BLEU-1 / ROUGE-L) come from:
      results/baselines/<mode>_metrics.json
    """
    metrics = {}
    for mode in ("zero_shot", "few_shot", "sft", "policy"):
        # 1. LLM absolute score (primary)
        for stem in (
            f"llm_eval_absolute_{mode}",   # when --output is named by mode
            f"llm_eval_absolute",          # default output name (single run)
        ):
            mpath = results_dir / f"{stem}_metrics.json"
            data = safe_load_json(mpath)
            if data and data.get("predictions_file", "").endswith(f"{mode}_predictions.jsonl"):
                metrics.setdefault(mode, {}).update(data)
                break

        # 2. Lexical metrics produced by run_baselines.py
        lex_path = results_dir / "baselines" / f"{mode}_metrics.json"
        lex_data = safe_load_json(lex_path)
        if lex_data:
            metrics.setdefault(mode, {}).update(lex_data)

    return metrics


def load_exp2(results_dir: Path) -> dict | None:
    return safe_load_json(results_dir / "exp2_human_eval.json")


def load_exp3(results_dir: Path) -> dict | None:
    return safe_load_json(results_dir / "exp3_ood" / "summary_table.json")


def load_exp1(results_dir: Path) -> dict | None:
    """Exp 1: RM agreement (Yosie's results). Load if available."""
    return safe_load_json(results_dir / "exp1_rm_agreement.json")


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

MODEL_LABELS = {
    "zero_shot": "Zero-shot (Qwen3-VL-2B-Thinking)",
    "few_shot":  "Few-shot (Qwen3-VL-2B-Thinking)",
    "sft":       "SFT (Qwen3-VL-2B-Thinking)",
    "policy":    "GRPO Policy (ours)",
    "tie":       "(Tie)",
}

PLACEHOLDER = "—"


def fmt(val, fmt_str=".2f", fallback=PLACEHOLDER):
    try:
        return format(float(val), fmt_str)
    except (TypeError, ValueError):
        return fallback


def build_main_table(llm_metrics: dict, exp2: dict | None, exp3: dict | None) -> list[dict]:
    """Master results table: one row per model."""
    models = ["zero_shot", "few_shot", "sft", "policy"]
    rows = []

    # LLM absolute scores (in-distribution)
    llm_scores = {}
    bleu1_scores = {}
    rougeL_scores = {}
    for mode, m in llm_metrics.items():
        llm_scores[mode] = m.get("mean_humor_score")
        bleu1_scores[mode] = m.get("bleu1")
        rougeL_scores[mode] = m.get("rougeL")

    # Exp 2: human eval win rates
    exp2_wr = {}
    if exp2:
        exp2_wr = exp2.get("win_rates", {}).get("win_rates", {})

    # Exp 3: OOD LLM scores
    exp3_scores = {}
    if exp3 and isinstance(exp3, list):
        for row in exp3:
            exp3_scores[row["model"]] = row.get("mean_score")

    for model in models:
        rows.append({
            "model": model,
            "label": MODEL_LABELS.get(model, model),
            "bleu1": bleu1_scores.get(model),
            "rougeL": rougeL_scores.get(model),
            "llm_humor_score_indist": llm_scores.get(model),
            "human_win_rate": exp2_wr.get(model),
            "llm_humor_score_ood": exp3_scores.get(model),
        })

    return rows


def render_main_latex(table_rows: list[dict]) -> str:
    lines = []
    for row in table_rows:
        label    = row["label"]
        bleu1    = fmt(row["bleu1"], ".3f")
        rougeL   = fmt(row["rougeL"], ".3f")
        score_id = fmt(row["llm_humor_score_indist"])
        win_rate = fmt(row["human_win_rate"], ".1%") if row["human_win_rate"] is not None else PLACEHOLDER
        score_ood = fmt(row["llm_humor_score_ood"])
        lines.append(
            f"  {label:<42} & {bleu1:>7} & {rougeL:>8} & {score_id:>8} & {win_rate:>10} & {score_ood:>8} \\\\\\"
        )

    rows_str = "\n".join(lines)
    return rf"""
\begin{{table}}[h]
\centering
\caption{{Main results. BLEU-1 and ROUGE-L are lexical overlap with ground-truth captions.
LLM Humor Score is the absolute humor rating (1--5) from the LLM judge.
Win Rate is human pairwise preference vs.\ other systems (Exp 2).
OOD Score is LLM humor score on out-of-distribution images (Exp 3).}}
\label{{tab:main-results}}
\begin{{tabular}}{{lccccc}}
\toprule
\textbf{{Model}} & \textbf{{BLEU-1}} & \textbf{{ROUGE-L}} & \textbf{{LLM Score (ID)}} & \textbf{{Human Win Rate}} & \textbf{{LLM Score (OOD)}} \\
\midrule
{rows_str}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def render_exp2_latex(exp2: dict | None) -> str:
    if not exp2:
        return "% Exp 2 data not yet available.\n"

    win_rates = exp2.get("win_rates", {}).get("win_rates", {})
    win_counts = exp2.get("win_rates", {}).get("model_win_counts", {})
    total = exp2.get("win_rates", {}).get("total_comparisons", 0)
    n_raters = exp2.get("rater_stats", {}).get("n_raters", PLACEHOLDER)
    agreement = exp2.get("inter_rater_agreement", {}).get("mean_agreement")

    rows = []
    for model, wr in sorted(win_rates.items(), key=lambda x: -x[1]):
        label = MODEL_LABELS.get(model, model)
        count = win_counts.get(model, 0)
        rows.append(
            f"  {label:<42} & {wr:.1%} & {count}/{total} \\\\"
        )
    rows_str = "\n".join(rows)

    agr_str = f"{agreement:.1%}" if agreement else PLACEHOLDER
    return rf"""
\begin{{table}}[h]
\centering
\caption{{Exp 2: Human evaluation of policy captions. Win rate = fraction of pairwise
comparisons won by each model. $N_\text{{raters}}$ = {n_raters}.
Inter-rater agreement = {agr_str}.}}
\label{{tab:exp2-human}}
\begin{{tabular}}{{lcc}}
\toprule
\textbf{{Model}} & \textbf{{Win Rate}} & \textbf{{Wins / Total}} \\
\midrule
{rows_str}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def render_exp3_latex(exp3: list | None) -> str:
    if not exp3:
        return "% Exp 3 data not yet available.\n"

    rows = []
    for row in sorted(exp3, key=lambda x: -(x.get("mean_score") or 0)):
        label = MODEL_LABELS.get(row["model"], row["model"])
        score = fmt(row.get("mean_score"))
        n = row.get("n", PLACEHOLDER)
        rows.append(f"  {label:<42} & {score:>8} & {n:>6} \\\\")
    rows_str = "\n".join(rows)

    return rf"""
\begin{{table}}[h]
\centering
\caption{{Exp 3: Out-of-distribution (OOD) evaluation. LLM Humor Score
rated by Claude / GPT-4o on 1--5 scale over diverse non-cartoon images.}}
\label{{tab:exp3-ood}}
\begin{{tabular}}{{lcc}}
\toprule
\textbf{{Model}} & \textbf{{LLM Score (OOD)}} & \textbf{{N}} \\
\midrule
{rows_str}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def render_markdown_summary(table_rows: list[dict], exp1: dict | None,
                             exp2: dict | None, exp3: list | None) -> str:
    lines = ["# HumorR1 Results Summary\n"]

    # Main table
    lines.append("## Main Results\n")
    lines.append(
        f"| {'Model':<42} | {'BLEU-1':>7} | {'ROUGE-L':>8} | {'LLM Score (ID)':>15} | {'Human Win Rate':>15} | {'LLM Score (OOD)':>16} |"
    )
    lines.append(
        f"|{'-'*43}|{'-'*8}|{'-'*9}|{'-'*16}|{'-'*16}|{'-'*17}|"
    )
    for row in table_rows:
        label  = row["label"]
        bleu1  = fmt(row["bleu1"], ".3f")
        rougeL = fmt(row["rougeL"], ".3f")
        s_id   = fmt(row["llm_humor_score_indist"])
        wr     = fmt(row["human_win_rate"], ".1%") if row["human_win_rate"] is not None else PLACEHOLDER
        s_ood  = fmt(row["llm_humor_score_ood"])
        lines.append(f"| {label:<42} | {bleu1:>7} | {rougeL:>8} | {s_id:>15} | {wr:>15} | {s_ood:>16} |")
    lines.append("")

    # Exp 1
    lines.append("## Exp 1: Reward Model Agreement\n")
    if exp1:
        acc = exp1.get("accuracy")
        lines.append(f"- RM human agreement accuracy: **{fmt(acc, '.1%')}**")
    else:
        lines.append("_Data not yet available (Yosie's results)._\n")
    lines.append("")

    # Exp 2
    lines.append("## Exp 2: Human Eval Win Rates\n")
    if exp2:
        wr_dict = exp2.get("win_rates", {}).get("win_rates", {})
        n_raters = exp2.get("rater_stats", {}).get("n_raters", PLACEHOLDER)
        lines.append(f"N raters: {n_raters}\n")
        lines.append("| Model | Win Rate |")
        lines.append("|-------|----------|")
        for model, wr in sorted(wr_dict.items(), key=lambda x: -x[1]):
            lines.append(f"| {MODEL_LABELS.get(model, model)} | {wr:.1%} |")
    else:
        lines.append("_Data not yet available (waiting on policy captions + survey responses)._\n")
    lines.append("")

    # Exp 3
    lines.append("## Exp 3: OOD Eval\n")
    if exp3:
        lines.append("| Model | LLM Score (OOD) | N |")
        lines.append("|-------|-----------------|---|")
        for row in sorted(exp3, key=lambda x: -(x.get("mean_score") or 0)):
            label = MODEL_LABELS.get(row["model"], row["model"])
            lines.append(f"| {label} | {fmt(row.get('mean_score'))} | {row.get('n', PLACEHOLDER)} |")
    else:
        lines.append("_Data not yet available (waiting on policy outputs + OOD curation)._\n")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Compile all experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tables"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    llm_metrics = load_llm_metrics(args.results_dir)
    exp1 = load_exp1(args.results_dir)
    exp2 = load_exp2(args.results_dir)
    exp3 = load_exp3(args.results_dir)

    print(f"  LLM metrics:   {list(llm_metrics.keys()) or '(none)'}")
    print(f"  Exp 1 (RM):    {'loaded' if exp1 else 'not found'}")
    print(f"  Exp 2 (human): {'loaded' if exp2 else 'not found'}")
    print(f"  Exp 3 (OOD):   {'loaded' if exp3 else 'not found'}")

    # Build tables
    table_rows = build_main_table(llm_metrics, exp2, exp3)

    # Save master table
    master_path = args.output_dir / "table_main.json"
    master_path.write_text(json.dumps(table_rows, indent=2))

    # LaTeX files
    (args.output_dir / "table_main.tex").write_text(render_main_latex(table_rows))
    (args.output_dir / "table_exp2.tex").write_text(render_exp2_latex(exp2))
    (args.output_dir / "table_exp3.tex").write_text(render_exp3_latex(exp3))

    # Markdown summary
    summary_md = render_markdown_summary(table_rows, exp1, exp2, exp3)
    (args.output_dir / "summary.md").write_text(summary_md)

    print(f"\nTables written to {args.output_dir}/")
    print(f"  table_main.json   — master machine-readable table")
    print(f"  table_main.tex    — LaTeX main results table")
    print(f"  table_exp2.tex    — LaTeX Exp 2 human eval table")
    print(f"  table_exp3.tex    — LaTeX Exp 3 OOD eval table")
    print(f"  summary.md        — Markdown summary\n")

    # Print markdown to stdout
    print(summary_md)


if __name__ == "__main__":
    main()
