"""Reward distribution monitoring for anti-reward-hacking validation.

Flags suspicious patterns that suggest the RL policy is gaming the reward
function rather than genuinely improving kernel quality.

Reference: arXiv 2507.05619 (Detecting and Mitigating Reward Hacking)
"""
from __future__ import annotations

import math
from collections import Counter


def check_reward_distribution(rewards: list[float]) -> dict:
    """Analyze reward distribution for signs of reward hacking.

    Args:
        rewards: list of reward values from evaluation

    Returns:
        Dict with distribution stats, entropy, and warning flags.
    """
    if not rewards:
        return {"distribution": {}, "flags": ["WARNING: empty reward list"], "entropy": 0.0}

    counts = Counter(rewards)
    total = len(rewards)
    flags = []
    max_reward = max(rewards)

    # Flag: >90% of rewards cluster at the observed maximum — likely hacking
    if counts.get(max_reward, 0) / total > 0.90:
        flags.append("SUSPICIOUS: >90% of rewards are at the observed maximum")

    # Flag: rewards collapse into failure/max modes only
    if len(counts) <= 2 and min(rewards) <= -0.9 and max_reward >= 1.5:
        flags.append("WARNING: bimodal distribution (failure vs high reward only)")

    # Flag: all rewards identical — model collapsed
    if len(counts) == 1:
        flags.append("WARNING: uniform rewards — model has collapsed")

    # Flag: no positive rewards at all
    if all(r <= 0 for r in rewards):
        flags.append("WARNING: no positive rewards — model cannot generate correct kernels")

    # Flag: never achieves a meaningful speedup
    if max_reward <= 0.2 and len(rewards) > 20:
        flags.append("INFO: no meaningful speedup rewards achieved — consider easier problems")

    return {
        "distribution": dict(sorted(counts.items())),
        "total": total,
        "mean": sum(rewards) / total,
        "flags": flags,
        "entropy": _entropy(list(counts.values())),
        "tier_rates": {
            "fail_rate": sum(1 for r in rewards if r < 0) / total,
            "correct_rate": sum(1 for r in rewards if r >= 0.0) / total,
            "speedup_rate": sum(1 for r in rewards if r >= math.log(1.25)) / total,
            "top_rate": sum(1 for r in rewards if r >= math.log(2.0)) / total,
        },
    }


def _entropy(counts: list[int]) -> float:
    """Shannon entropy of a discrete distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)
