"""Stub reward for GRPO pipeline smoke-testing.

Binary format reward: 1.0 if the completion contains a well-formed
`<caption>...</caption>` block with non-empty content, else 0.0. The
whole point is to confirm the RL loop can move reward upward on a
trivial objective. Replace with the trained Bradley-Terry reward model
(Qwen3-VL on New Yorker preference pairs) once that's ready.

The train entrypoint (scripts/train_grpo.py) wraps this in a TRL-compatible
``reward_fn(completions, **kwargs) -> list[float]`` shim.
"""

import re

_CAPTION_RE = re.compile(r"<caption>(.*?)</caption>", re.DOTALL)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    match = _CAPTION_RE.search(solution_str or "")
    if match and match.group(1).strip():
        return 1.0
    return 0.0
