"""Reward helpers: continuous log(speedup) + Nsight bonus + TRLOO post-process.

Replaces discrete {-1,1,2,3} milestone scheme (Fix 5).
Reuses CUDA-Agent's profiling.py + verification.py via subprocess.
"""
from __future__ import annotations

import math

_REQUIRED_EVAL_KEYS = {"compiles", "correct", "speedup_vs_orig", "speedup_vs_dg", "error"}


def validate_eval_result(result: dict) -> dict:
    """Validate Modal evaluate_kernel return schema. Missing/invalid → safe defaults."""
    missing = _REQUIRED_EVAL_KEYS - set(result)
    if missing:
        return {"compiles": False, "correct": False, "speedup_vs_orig": 0.0,
                "speedup_vs_dg": 0.0, "error": f"missing keys: {missing}"}
    out = dict(result)
    # Clamp NaN/inf speedups to 0
    for k in ("speedup_vs_orig", "speedup_vs_dg"):
        v = out.get(k, 0.0)
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            out[k] = 0.0
    return out


def compute_reward(
    compiled: bool,
    correct: bool,
    speedup_vs_eager: float,
    speedup_vs_compile: float,
    occupancy: float | None = None,
    mem_coalescing: float | None = None,
    warp_efficiency: float | None = None,
) -> float:
    """Return continuous reward based on log(speedup) + Nsight bonus.

    Args:
        compiled: Whether the kernel compiled successfully.
        correct: Whether the kernel produces correct output.
        speedup_vs_eager: Speedup ratio vs torch.eager baseline.
        speedup_vs_compile: Speedup ratio vs torch.compile baseline.
        occupancy: SM occupancy from Nsight/profiling (0.0-1.0), or None.
        mem_coalescing: Memory coalescing efficiency (0.0-1.0), or None.
        warp_efficiency: Warp execution efficiency (0.0-1.0), or None.

    Returns:
        -1.0 for compile/correctness failure (REGARDLESS of speedup — correctness
        is checked BEFORE speedup to prevent reward hacking).
        log(speedup) + nsight_bonus for correct kernels.
    """
    # Correctness gate: must pass BEFORE any speedup signal reaches gradients.
    # A fast-but-wrong kernel MUST get -1.0, not a positive reward.
    if not compiled or not correct:
        return -1.0

    # Continuous speedup signal (log scale for proportional gradient)
    base = math.log(max(speedup_vs_eager, 0.1))

    # Nsight bonus when profiling metrics are available
    if occupancy is not None:
        occ = max(0.0, min(1.0, occupancy))
        mem = max(0.0, min(1.0, mem_coalescing or 0.0))
        warp = max(0.0, min(1.0, warp_efficiency or 0.0))
        base += 0.4 * occ + 0.3 * mem + 0.2 * warp

    return base


def trloo_post_process(advantages: list[float], n: int) -> list[float]:
    """Scale GRPO advantages by N/(N-1) to correct gradient shrinkage.

    Dr. Kernel (arXiv 2602.05885) proves GRPO's self-inclusion bias
    shrinks expected gradients by (1 - 1/N). With G=4, that is 25%.
    This post-process is a drop-in fix for TRL GRPOTrainer.
    """
    if n <= 1:
        return advantages
    scale = n / (n - 1)
    return [a * scale for a in advantages]
