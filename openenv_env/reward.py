"""Reward helpers: continuous log(speedup) + Nsight bonus + TRLOO post-process.

Replaces discrete {-1,1,2,3} milestone scheme (Fix 5).
Reuses CUDA-Agent's profiling.py + verification.py via subprocess.
"""
from __future__ import annotations

import math


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
        -1.0 for compile/correctness failure.
        log(speedup) + nsight_bonus for correct kernels.
    """
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
