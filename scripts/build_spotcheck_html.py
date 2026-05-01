"""Build a self-contained HTML page for the 3-rater human spot-check.

Each page loads 20 random test cartoons (seeded). For each cartoon, all 7
cells' captions are shown in a randomized, anonymized order; the rater
ranks them 1 (funniest) to 7 (least funny) by typing into 7 numeric
inputs. Submit produces a JSON object the rater pastes into a shared
file (or a `?download=` link).

Usage:
    uv run python scripts/build_spotcheck_html.py \\
        --out paper/spotcheck.html \\
        --n-cartoons 20 \\
        --seed 42

After the three teammates each fill it out, run:
    uv run python scripts/score_spotcheck.py
to compute Krippendorff alpha and rank-correlation with judge/RM.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CELLS = ["E0a", "E0b", "E0c", "E1a", "E1b", "E2a", "E2b"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "paper" / "spotcheck.html")
    p.add_argument("--captions-dir", type=Path,
                   default=PROJECT_ROOT / "results" / "captions")
    p.add_argument("--n-cartoons", type=int, default=20)
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-side", type=int, default=512)
    return p.parse_args()


def load_cell_captions(path: Path) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("caption"):
            out.setdefault(int(r["contest_number"]), []).append(r["caption"])
    return out


def encode_image(path: Path, max_side: int) -> str:
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    # Collect captions for each cell on the chosen split
    cell_caps: dict[str, dict[int, list[str]]] = {}
    image_paths: dict[int, str] = {}
    for cell in CELLS:
        path = args.captions_dir / f"{cell}_{args.split}.jsonl"
        cell_caps[cell] = load_cell_captions(path)
        # Also pull image_paths
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            image_paths.setdefault(int(r["contest_number"]), r["image_path"])

    # Common contests across all cells
    common = sorted(set.intersection(*(set(d.keys()) for d in cell_caps.values())))
    rng.shuffle(common)
    selected = common[: args.n_cartoons]
    print(f"Selected {len(selected)} contests: {selected}")

    # For each contest, pick one random caption per cell, randomize positions
    items = []
    for c in selected:
        picks = []
        for cell in CELLS:
            caps = cell_caps[cell].get(c, [])
            if not caps:
                continue
            picks.append({"cell": cell, "caption": rng.choice(caps)})
        rng.shuffle(picks)
        # Anonymize cell IDs (so rater can't bias)
        for i, p in enumerate(picks):
            p["display_id"] = chr(ord("A") + i)
        img_b64 = encode_image(Path(image_paths[c]), args.max_side)
        items.append({
            "contest_number": c,
            "image_b64": img_b64,
            "candidates": picks,
        })

    # Build HTML
    html_parts = ["""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>humor-r1 spot-check</title>
<style>
body { font-family: sans-serif; max-width: 920px; margin: 30px auto; padding: 0 16px; }
h1 { border-bottom: 2px solid #444; padding-bottom: 6px; }
.cartoon { border: 1px solid #ccc; padding: 16px; margin: 24px 0; border-radius: 8px; background: #fafafa; }
img { max-width: 100%; max-height: 480px; border: 1px solid #888; }
.cap { margin: 8px 0; padding: 10px 14px; background: #fff; border-left: 4px solid #ddd; border-radius: 4px; font-size: 16px; }
.cap input { width: 50px; margin-right: 12px; padding: 4px; font-size: 14px; text-align: center; }
.controls { position: sticky; top: 0; background: #fff; padding: 10px; border-bottom: 1px solid #ccc; }
button { padding: 8px 16px; font-size: 16px; background: #2962ff; color: #fff; border: none; border-radius: 4px; cursor: pointer; }
button:hover { background: #1e44b3; }
textarea { width: 100%; height: 200px; font-family: monospace; }
.warn { color: #c00; }
</style>
</head>
<body>
<h1>humor-r1 — caption ranking spot-check</h1>
<p>For each cartoon below, rank the captions from <b>1 (funniest)</b> to <b>7 (least funny)</b> by typing a number 1–7 in front of each caption. Use each number exactly once per cartoon (no ties). When done, click <b>Compute</b> and paste the JSON into a shared file along with your name.</p>
<div class="controls">
  <label>Your name: <input type="text" id="rater_name" /></label>
  <button onclick="compute()">Compute JSON</button>
  <span id="status"></span>
</div>
"""]
    for idx, item in enumerate(items):
        html_parts.append(f"""
<div class="cartoon" data-contest="{item['contest_number']}" data-idx="{idx}">
<h3>Cartoon #{idx+1} (contest {item['contest_number']})</h3>
<img src="data:image/jpeg;base64,{item['image_b64']}" />
<p>Rank these captions 1–7:</p>
""")
        for p in item["candidates"]:
            html_parts.append(f"""
<div class="cap"><input type="number" min="1" max="7" data-cell="{p['cell']}" data-display="{p['display_id']}" /> <b>{p['display_id']}.</b> {p['caption']}</div>
""")
        html_parts.append("</div>")

    html_parts.append("""
<h2>Submit</h2>
<p>JSON output (paste into a file named <code>spotcheck_&lt;your_initials&gt;.json</code>):</p>
<textarea id="output" readonly></textarea>
<script>
function compute() {
    const rater = document.getElementById('rater_name').value.trim();
    if (!rater) { alert('Please enter your name first.'); return; }
    const cartoons = document.querySelectorAll('.cartoon');
    const out = { rater: rater, n_cartoons: cartoons.length, items: [] };
    let issues = [];
    for (const cart of cartoons) {
        const contest = cart.dataset.contest;
        const ranks = {};
        const seen = {};
        for (const inp of cart.querySelectorAll('input[data-cell]')) {
            const r = parseInt(inp.value);
            if (!r || r < 1 || r > 7) {
                issues.push(`Cartoon ${contest}: missing or invalid rank for ${inp.dataset.display}`);
            }
            if (seen[r]) {
                issues.push(`Cartoon ${contest}: duplicate rank ${r}`);
            }
            seen[r] = true;
            ranks[inp.dataset.cell] = r;
        }
        out.items.push({ contest_number: parseInt(contest), ranks: ranks });
    }
    document.getElementById('output').value = JSON.stringify(out, null, 2);
    const status = document.getElementById('status');
    if (issues.length) {
        status.innerHTML = `<span class="warn">${issues.length} issue(s): ${issues.slice(0,3).join('; ')}${issues.length > 3 ? '...' : ''}</span>`;
    } else {
        status.innerText = `Looks complete (${cartoons.length} cartoons, ${rater}).`;
    }
}
</script>
</body></html>
""")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(html_parts))
    print(f"Wrote {args.out} ({args.out.stat().st_size//1024} KB) with {len(items)} cartoons")
    print(f"Each rater opens this file in a browser, ranks, and pastes JSON output.")


if __name__ == "__main__":
    main()
