"""Best-of-N reranking analysis using existing scored captions.

For each cartoon and each cell with N=5 samples, computes:
- mean-of-5 RM (what the main results table shows)
- best-of-5 RM (BoN with our RM as the selector)
- worst-of-5 RM (lower bound)

Intuition: if the RM is even modestly informative, BoN should beat
mean-of-5 (and importantly, BoN of a small-model should approach the
single-sample frontier baseline).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import statistics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CELLS = ["E0a", "E0b", "E0c", "E1a", "E1b", "E2a", "E2b"]


def load_scored(cell: str, split: str) -> dict[int, list[float]]:
    """contest_number -> [rm_score, ...] over the cell's samples."""
    path = PROJECT_ROOT / "results" / "captions" / f"{cell}_{split}.scored.jsonl"
    if not path.exists():
        return {}
    out: dict[int, list[float]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("rm_score") is None:
            continue
        out[int(r["contest_number"])].append(float(r["rm_score"]))
    return dict(out)


def summarize() -> None:
    rows = []
    for split in ("validation", "test"):
        for cell in CELLS:
            scores = load_scored(cell, split)
            mean_per_cartoon = []
            best_per_cartoon = []
            worst_per_cartoon = []
            n_per_cartoon = []
            for c, scs in scores.items():
                if len(scs) < 1:
                    continue
                mean_per_cartoon.append(statistics.mean(scs))
                best_per_cartoon.append(max(scs))
                worst_per_cartoon.append(min(scs))
                n_per_cartoon.append(len(scs))
            if not mean_per_cartoon:
                continue
            rows.append({
                "cell": cell,
                "split": split,
                "n_cartoons": len(mean_per_cartoon),
                "n_per_cartoon": (statistics.mean(n_per_cartoon)
                                  if n_per_cartoon else 0),
                "mean_of_N": statistics.mean(mean_per_cartoon),
                "best_of_N": statistics.mean(best_per_cartoon),
                "worst_of_N": statistics.mean(worst_per_cartoon),
                "best_minus_mean": (statistics.mean(best_per_cartoon)
                                     - statistics.mean(mean_per_cartoon)),
            })

    print(f"{'cell':<5} {'split':<11} {'n':>3} {'N':>3}  {'worst':>7}  {'mean':>7}  "
          f"{'best':>7}  {'BoN-mean':>9}")
    for r in rows:
        print(f"{r['cell']:<5} {r['split']:<11} {r['n_cartoons']:>3} "
              f"{r['n_per_cartoon']:>3.0f}  {r['worst_of_N']:>+7.3f}  "
              f"{r['mean_of_N']:>+7.3f}  {r['best_of_N']:>+7.3f}  "
              f"{r['best_minus_mean']:>+9.3f}")

    out_path = PROJECT_ROOT / "results" / "best_of_n.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    summarize()
