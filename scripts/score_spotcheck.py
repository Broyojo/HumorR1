"""Aggregate human spot-check JSONs (one per rater), compute Krippendorff
alpha and rank correlation against the Sonnet judge BT and the RM mean.

Inputs:
  paper/spotcheck_*.json  (one per rater, produced by spotcheck.html)
  results/numbers.json    (judge + RM scores from compile_paper_tables.py)

Outputs:
  results/spotcheck_summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CELLS = ["E0a", "E0b", "E0c", "E1a", "E1b", "E2a", "E2b", "E3"]


def krippendorff_alpha_ordinal(matrix: np.ndarray) -> float:
    """Krippendorff alpha for ordinal/interval data.

    matrix: (n_raters, n_units), with NaN for missing.
    """
    # Flatten to (rater, unit, value); use interval distance.
    n_raters, n_units = matrix.shape
    # Pairs of rated values per unit
    Du = []
    De_pairs = []
    for u in range(n_units):
        col = matrix[:, u]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            continue
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                Du.append((valid[i] - valid[j]) ** 2)
        for v in valid:
            De_pairs.append(v)
    Du_mean = np.mean(Du) if Du else 0.0
    # Expected disagreement = mean over all pairs across units
    De = []
    for i in range(len(De_pairs)):
        for j in range(i + 1, len(De_pairs)):
            De.append((De_pairs[i] - De_pairs[j]) ** 2)
    De_mean = np.mean(De) if De else 0.0
    if De_mean == 0:
        return float("nan")
    return 1.0 - Du_mean / De_mean


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rater-glob", default="paper/spotcheck_*.json")
    parser.add_argument("--out", default="results/spotcheck_summary.json")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.rater_glob))
    if not paths:
        print(f"No spot-check files matched {args.rater_glob}", file=sys.stderr)
        return 1
    print(f"Found {len(paths)} rater files: {[Path(p).name for p in paths]}")

    # Build matrix: ranking per (rater, unit=cartoon×cell)
    contests: list[int] = []
    rater_data: list[dict[tuple[int, str], int]] = []
    raters: list[str] = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        raters.append(d["rater"])
        ranks_by_unit = {}
        for it in d["items"]:
            c = int(it["contest_number"])
            for cell, r in it["ranks"].items():
                ranks_by_unit[(c, cell)] = int(r)
            if c not in contests:
                contests.append(c)
        rater_data.append(ranks_by_unit)

    units = [(c, cell) for c in contests for cell in CELLS]
    matrix = np.full((len(raters), len(units)), np.nan)
    for r_i, rd in enumerate(rater_data):
        for u_i, u in enumerate(units):
            if u in rd:
                matrix[r_i, u_i] = rd[u]

    alpha = krippendorff_alpha_ordinal(matrix)

    # Mean rank per cell across all raters and cartoons (lower = funnier)
    cell_means = {}
    for cell in CELLS:
        ranks = []
        for r_i in range(len(raters)):
            for c in contests:
                idx = units.index((c, cell))
                v = matrix[r_i, idx]
                if not math.isnan(v):
                    ranks.append(v)
        cell_means[cell] = float(np.mean(ranks)) if ranks else float("nan")

    # Compare to judge BT and RM mean (test split)
    with open(PROJECT_ROOT / "results" / "numbers.json") as f:
        numbers = json.load(f)
    judge_bt = {c: numbers["judge_bt_test"].get(c) for c in CELLS}
    rm_test = {c: (numbers["rm"].get(f"{c}|test") or {}).get("mean") for c in CELLS}

    # Spearman: human mean rank (lower = better) vs -judge_bt and -rm (negated so lower=better)
    cells_with_data = [c for c in CELLS if not math.isnan(cell_means[c])
                       and judge_bt.get(c) is not None and rm_test.get(c) is not None]
    if len(cells_with_data) < 3:
        print("Not enough data for correlation", file=sys.stderr)
        return 1
    human_ranks_array = np.array([cell_means[c] for c in cells_with_data])
    judge_array = np.array([-judge_bt[c] for c in cells_with_data])
    rm_array = np.array([-rm_test[c] for c in cells_with_data])
    rho_judge, p_judge = spearmanr(human_ranks_array, judge_array)
    rho_rm, p_rm = spearmanr(human_ranks_array, rm_array)

    summary = {
        "n_raters": len(raters),
        "raters": raters,
        "n_cartoons": len(contests),
        "krippendorff_alpha": alpha,
        "human_mean_rank_per_cell": cell_means,
        "spearman_human_vs_judge": {"rho": float(rho_judge), "p": float(p_judge)},
        "spearman_human_vs_rm": {"rho": float(rho_rm), "p": float(p_rm)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\n=== Spot-check summary ===")
    print(f"raters: {raters}")
    print(f"n cartoons: {len(contests)}")
    print(f"Krippendorff alpha (ordinal): {alpha:.3f}")
    print(f"Human mean rank per cell (lower=funnier):")
    for cell in CELLS:
        print(f"  {cell}: {cell_means[cell]:.2f}")
    print(f"Spearman vs Sonnet BT: rho={rho_judge:.3f} p={p_judge:.4f}")
    print(f"Spearman vs RM:        rho={rho_rm:.3f} p={p_rm:.4f}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
