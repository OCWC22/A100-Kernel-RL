"""
Curriculum learning for progressive CUDA kernel difficulty.

4 phases based on DoubleGraph's 4-way dispatch pattern:
  1. single_ops    — single operator kernels (vector_add, relu, softmax)
  2. fusion_2op    — 2-op fusions (83.77% of Ops-6K, the sweet spot)
  3. arch_specific — A100-specific optimizations (L2 pinning, cooperative kernels)
  4. advanced      — multi-op fusion, complex memory patterns

Promotion: >50% of last 10 rewards hit target → advance
Demotion:  <20% of last 10 rewards are positive (>0) → regress
"""
from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass, field

TARGET_GPU = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")

WINDOW_SIZE = 10
PROMOTE_THRESHOLD = 0.50   # >50% hit target reward → advance
DEMOTE_THRESHOLD = 0.20    # <20% positive → regress


@dataclass
class CurriculumPhase:
    """A single phase of the curriculum."""
    name: str
    target_reward: float
    problems: list[dict] = field(default_factory=list)
    description: str = ""


def _default_phases() -> list[CurriculumPhase]:
    """Build the 4-phase curriculum with built-in fallback problems."""
    return [
        CurriculumPhase(
            name="single_ops",
            target_reward=1.0,
            description="Single operator kernels — establish CUDA syntax and correctness",
            problems=[
                {
                    "prompt": f"Write a CUDA kernel for vector addition optimized for {TARGET_GPU} ({TARGET_ARCH}). "
                              "Take float arrays A, B, output C, and length N.",
                    "ops": ["vector_add"],
                    "difficulty": 1,
                },
                {
                    "prompt": f"Write a CUDA ReLU activation kernel for {TARGET_GPU} ({TARGET_ARCH}). "
                              "Apply max(0, x) element-wise to a float array.",
                    "ops": ["relu"],
                    "difficulty": 1,
                },
                {
                    "prompt": f"Write a CUDA softmax kernel for {TARGET_GPU} ({TARGET_ARCH}). "
                              "Compute row-wise softmax for a 2D float matrix.",
                    "ops": ["softmax"],
                    "difficulty": 1,
                },
                {
                    "prompt": f"Write a CUDA matrix multiplication kernel for {TARGET_GPU} ({TARGET_ARCH}). "
                              "Compute C = A @ B for float matrices using shared memory tiling.",
                    "ops": ["matmul"],
                    "difficulty": 1,
                },
                {
                    "prompt": f"Write a CUDA kernel for element-wise GELU activation for {TARGET_GPU} ({TARGET_ARCH}).",
                    "ops": ["gelu"],
                    "difficulty": 1,
                },
            ],
        ),
        CurriculumPhase(
            name="fusion_2op",
            target_reward=2.0,
            description="2-op fusions — the Ops-6K sweet spot (83.77% of dataset)",
            problems=[
                {
                    "prompt": f"Write a fused CUDA kernel for LayerNorm + ReLU for {TARGET_GPU} ({TARGET_ARCH}). "
                              "Combine normalization and activation in a single kernel launch.",
                    "ops": ["layer_norm", "relu"],
                    "difficulty": 2,
                },
                {
                    "prompt": f"Write a fused CUDA kernel for MatMul + BiasAdd for {TARGET_GPU} ({TARGET_ARCH}). "
                              "Compute C = A @ B + bias in a single kernel.",
                    "ops": ["matmul", "bias_add"],
                    "difficulty": 2,
                },
                {
                    "prompt": f"Write a fused CUDA kernel for Softmax + Dropout for {TARGET_GPU} ({TARGET_ARCH}). "
                              "Apply softmax then probabilistic dropout in one pass.",
                    "ops": ["softmax", "dropout"],
                    "difficulty": 2,
                },
                {
                    "prompt": f"Write a fused CUDA kernel for Conv2d + BatchNorm for {TARGET_GPU} ({TARGET_ARCH}). "
                              "Fuse the convolution output directly into batch normalization.",
                    "ops": ["conv2d", "batch_norm"],
                    "difficulty": 2,
                },
            ],
        ),
        CurriculumPhase(
            name="arch_specific",
            target_reward=2.0,
            description="A100-specific optimizations — L2 pinning, cooperative kernels, vectorized loads",
            problems=[
                {
                    "prompt": f"Write a CUDA WCC (Weakly Connected Components) kernel for {TARGET_GPU} ({TARGET_ARCH}) "
                              "using non-atomic Union-Find with L2 cache pinning for the parent array.",
                    "ops": ["wcc"],
                    "difficulty": 3,
                },
                {
                    "prompt": f"Write a CUDA reduction kernel for {TARGET_GPU} ({TARGET_ARCH}) using "
                              "cooperative groups grid-wide sync for single-kernel-launch reduction.",
                    "ops": ["reduce_sum"],
                    "difficulty": 3,
                },
                {
                    "prompt": f"Write a CUDA GEMM kernel for {TARGET_GPU} ({TARGET_ARCH}) using "
                              "shared memory double-buffering and vectorized float4 loads for maximum throughput.",
                    "ops": ["gemm"],
                    "difficulty": 3,
                },
                {
                    "prompt": f"Write a CUDA attention kernel for {TARGET_GPU} ({TARGET_ARCH}) that tiles "
                              "Q, K, V in shared memory to minimize HBM traffic. Target occupancy >75%.",
                    "ops": ["attention"],
                    "difficulty": 3,
                },
            ],
        ),
        CurriculumPhase(
            name="advanced",
            target_reward=3.0,
            description="Multi-op fusion and complex memory patterns — beat torch.compile",
            problems=[
                {
                    "prompt": f"Write a fused CUDA kernel for LayerNorm + GELU + Linear for {TARGET_GPU} ({TARGET_ARCH}). "
                              "Fuse all three operations to minimize memory round-trips.",
                    "ops": ["layer_norm", "gelu", "linear"],
                    "difficulty": 4,
                },
                {
                    "prompt": f"Write a fused Flash-Attention-style kernel for {TARGET_GPU} ({TARGET_ARCH}) that "
                              "computes Q @ K^T, applies causal mask, softmax, and V multiply in one kernel.",
                    "ops": ["flash_attention"],
                    "difficulty": 4,
                },
                {
                    "prompt": f"Write a CUDA kernel for batched sparse matrix-vector multiplication (SpMV) "
                              f"for {TARGET_GPU} ({TARGET_ARCH}), using CSR format with warp-level load balancing.",
                    "ops": ["batched_spmv"],
                    "difficulty": 4,
                },
            ],
        ),
    ]


class CurriculumManager:
    """Manages progressive difficulty for CUDA kernel training."""

    def __init__(self, phases: list[CurriculumPhase] | None = None):
        self.phases = phases or _default_phases()
        self.current_phase_idx = 0
        self.reward_history: deque[float] = deque(maxlen=WINDOW_SIZE)
        self.phase_history: list[dict] = []  # track promotions/demotions

    @property
    def current_phase(self) -> CurriculumPhase:
        return self.phases[self.current_phase_idx]

    @property
    def phase_name(self) -> str:
        return self.current_phase.name

    def get_problem(self) -> dict:
        """Sample a random problem from the current phase."""
        problems = self.current_phase.problems
        if not problems:
            return {"prompt": f"Write a CUDA kernel optimized for {TARGET_GPU} ({TARGET_ARCH}).", "ops": [], "difficulty": 1}
        return random.choice(problems)

    def record_reward(self, reward: float) -> str | None:
        """Record a reward and check for promotion/demotion.

        Returns:
            "promoted", "demoted", or None if no change.
        """
        self.reward_history.append(reward)

        if len(self.reward_history) < WINDOW_SIZE:
            return None

        rewards = list(self.reward_history)
        target = self.current_phase.target_reward

        # Check promotion: >50% of recent rewards hit target
        hit_rate = sum(1 for r in rewards if r >= target) / len(rewards)
        if hit_rate > PROMOTE_THRESHOLD and self.current_phase_idx < len(self.phases) - 1:
            self._advance()
            return "promoted"

        # Check demotion: <20% of recent rewards are positive
        positive_rate = sum(1 for r in rewards if r > 0) / len(rewards)
        if positive_rate < DEMOTE_THRESHOLD and self.current_phase_idx > 0:
            self._regress()
            return "demoted"

        return None

    def _advance(self):
        old = self.current_phase.name
        self.current_phase_idx += 1
        self.reward_history.clear()
        self.phase_history.append({
            "action": "promoted",
            "from": old,
            "to": self.current_phase.name,
        })
        print(f"Curriculum: PROMOTED {old} → {self.current_phase.name}")

    def _regress(self):
        old = self.current_phase.name
        self.current_phase_idx -= 1
        self.reward_history.clear()
        self.phase_history.append({
            "action": "demoted",
            "from": old,
            "to": self.current_phase.name,
        })
        print(f"Curriculum: DEMOTED {old} → {self.current_phase.name}")

    def add_problems(self, phase_name: str, problems: list[dict]):
        """Add external problems (e.g. from Ops-6K) to a phase."""
        for phase in self.phases:
            if phase.name == phase_name:
                phase.problems.extend(problems)
                return
        raise ValueError(f"Unknown phase: {phase_name}")

    def status(self) -> dict:
        """Return current curriculum status."""
        rewards = list(self.reward_history)
        target = self.current_phase.target_reward
        return {
            "phase": self.current_phase.name,
            "phase_idx": self.current_phase_idx,
            "total_phases": len(self.phases),
            "target_reward": target,
            "window_size": len(rewards),
            "hit_rate": sum(1 for r in rewards if r >= target) / len(rewards) if rewards else 0,
            "positive_rate": sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0,
            "transitions": len(self.phase_history),
        }
