"""Stub reward for GRPO pipeline smoke-testing.

Returns a random score in [0, 1] so GRPO's advantage estimator sees nonzero
variance within each rollout group. Replace with the trained Bradley-Terry
reward model (Qwen3-VL on New Yorker preference pairs) once that's ready.

The train entrypoint (scripts/train_grpo.py) wraps this in a TRL-compatible
``reward_fn(completions, **kwargs) -> list[float]`` shim. Keeping the
per-sample scoring logic here means swapping to the real RM is a one-line
change in the training script.
"""

import random


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return random.random()
