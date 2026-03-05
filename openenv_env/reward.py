"""Reward helpers aligned with CUDA-Agent milestone scheme."""

from __future__ import annotations


def compute_reward(
    compiled: bool,
    correct: bool,
    speedup_vs_eager: float,
    speedup_vs_compile: float,
) -> float:
    """Return discrete milestone reward.

    Scheme:
    - -1.0: compile or correctness failure
    - +1.0: correct but no meaningful speedup
    - +2.0: >5% faster than eager baseline
    - +3.0: >5% faster than both eager and compile baselines
    """
    if not compiled or not correct:
        return -1.0
    if speedup_vs_compile > 1.05:
        return 3.0
    if speedup_vs_eager > 1.05:
        return 2.0
    return 1.0
