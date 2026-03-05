"""
Multi-turn rollout for GRPOTrainer — the core agentic loop.

Implements TRL's rollout_func pattern (docs: https://huggingface.co/docs/trl/main/en/openenv)
following the Wordle multi-turn example: loop over turns, call generate_rollout_completions()
per turn, step environment, accumulate token IDs + logprobs across turns.

See docs/TRAINING_PLAN.md for full rationale and architecture.
"""
from __future__ import annotations

import os
import re
from typing import Any, Callable

from openenv_env.reward import compute_reward

MODAL_APP_NAME = os.getenv("KERNELFORGE_MODAL_APP", "kernelforge-a100")


def extract_cuda_code(text: str) -> str:
    """Extract CUDA code from model output (fenced block or raw __global__)."""
    for marker in ["```cuda", "```cpp", "```c", "```c++"]:
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()

    if re.search(r"__global__\s+void\s+\w+", text):
        return text.strip()
    return ""


def _evaluate_on_modal(
    code: str,
    baseline_orig_ms: float | None = None,
    baseline_dg_ms: float | None = None,
) -> dict:
    """Send a single kernel to Modal for compile + verify + benchmark."""
    import modal

    eval_fn = modal.Function.from_name(MODAL_APP_NAME, "evaluate_kernel")
    return eval_fn.remote({
        "cuda_code": code,
        "verify_graphs": 5,
        "warmup_iters": 50,
        "benchmark_runs": 30,
        "baseline_original_ms": baseline_orig_ms,
        "baseline_doublegraph_ms": baseline_dg_ms,
    })


def _compute_reward_from_result(result: dict) -> float:
    """Compute discrete reward from Modal evaluation result."""
    return compute_reward(
        compiled=result.get("compiles", False),
        correct=result.get("correct", False),
        speedup_vs_eager=result.get("speedup_vs_orig", 0),
        speedup_vs_compile=result.get("speedup_vs_dg", 0),
    )


def _format_feedback(result: dict, reward: float, turn: int) -> str:
    """Format evaluation feedback for the model's next turn.

    Provides actionable information: exact error messages for compilation
    failures, invariant violations for verification failures, and performance
    metrics with optimization guidance for correct kernels.
    """
    parts = [f"[Turn {turn + 1} Result]"]

    if not result.get("compiles"):
        error = result.get("error", "unknown compilation error")
        parts.append(f"COMPILATION FAILED:\n{error[:800]}")
        parts.append("Fix the compilation errors above and resubmit.")
    elif not result.get("correct"):
        msg = result.get("verifier_msg", "unknown verification failure")
        parts.append(f"VERIFICATION FAILED: {msg}")
        parts.append("Your kernel produces incorrect output. Fix the algorithm.")
    else:
        runtime = result.get("runtime_ms", 0)
        speedup_eager = result.get("speedup_vs_orig", 0)
        speedup_compile = result.get("speedup_vs_dg", 0)
        stats = result.get("runtime_stats", {})
        parts.append(f"CORRECT. Runtime: {runtime:.3f}ms")
        parts.append(f"  Speedup vs eager: {speedup_eager:.2f}x")
        if speedup_compile:
            parts.append(f"  Speedup vs torch.compile: {speedup_compile:.2f}x")
        if stats:
            parts.append(
                f"  Stats: mean={stats.get('mean', 0):.3f}ms, "
                f"std={stats.get('std', 0):.3f}ms"
            )

        if reward < 2.0:
            parts.append(
                "Kernel is correct but not faster than eager PyTorch. "
                "Try shared memory tiling, vectorized loads, or kernel fusion."
            )
        elif reward < 3.0:
            parts.append(
                "Beats eager baseline! Try to also beat torch.compile. "
                "Consider L2 cache pinning, warp-level primitives, or "
                "register pressure tuning."
            )

    return "\n".join(parts)


_baselines_cache: dict | None = None


def _get_baselines() -> tuple[float | None, float | None]:
    """Fetch baseline timings from Modal (cached across calls)."""
    global _baselines_cache
    if _baselines_cache is None:
        try:
            import modal

            fn = modal.Function.from_name(MODAL_APP_NAME, "profile_baselines")
            _baselines_cache = fn.remote() or {}
        except Exception as e:
            print(f"Baseline profiling failed: {e}")
            _baselines_cache = {}
    return _baselines_cache.get("original_ms"), _baselines_cache.get("doublegraph_ms")


def make_multi_turn_rollout(
    max_turns: int = 5,
    skill_md_gpu: str | None = None,
) -> Callable:
    """Create a rollout_func for GRPOTrainer with multi-turn kernel refinement.

    Following TRL's OpenEnv Wordle pattern:
    - Loop over turns, calling generate_rollout_completions() per turn
    - Step environment (Modal evaluation) with each completion
    - Accumulate prompt_ids + completion_ids + logprobs across turns
    - Return best_reward per episode via extra dict fields

    Args:
        max_turns: Maximum refinement turns per episode (3 for warm-up, 5 for main RL)
        skill_md_gpu: GPU name for SKILL.md context (default: from env var)
    """
    from trl.experimental.openenv import generate_rollout_completions

    gpu_name = skill_md_gpu or os.getenv("KERNELFORGE_TARGET_GPU", "a100").lower()

    def rollout_func(prompts: list[str], trainer: Any) -> dict:
        from openenv_env.skill_builder import build_skill_md

        tokenizer = trainer.processing_class
        skill_context = build_skill_md(gpu_name)
        baseline_orig, baseline_dg = _get_baselines()

        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        all_best_rewards: list[float] = []

        for prompt_idx, initial_prompt in enumerate(prompts):
            episode_prompt_ids: list[int] = []
            episode_completion_ids: list[int] = []
            episode_logprobs: list[float] = []
            best_reward = -1.0

            # Build initial prompt with SKILL.md context
            current_prompt = (
                f"{skill_context}\n\n"
                f"---\n\n"
                f"{initial_prompt}\n\n"
                f"Write a complete CUDA kernel. Use ```cuda fenced code blocks."
            )

            for turn in range(max_turns):
                # Generate one completion via vLLM
                outputs = generate_rollout_completions(trainer, [current_prompt])[0]

                episode_prompt_ids.extend(outputs["prompt_ids"])
                episode_completion_ids.extend(outputs["completion_ids"])
                episode_logprobs.extend(outputs["logprobs"])

                # Decode completion text
                completion_text = outputs.get("text") or tokenizer.decode(
                    outputs["completion_ids"], skip_special_tokens=True
                )

                # Extract CUDA code
                code = extract_cuda_code(completion_text)
                if not code:
                    feedback = (
                        f"[Turn {turn + 1} Result]\n"
                        f"No valid CUDA code found in your response. "
                        f"Write a complete kernel with __global__ void function "
                        f"inside ```cuda fenced code blocks."
                    )
                    current_prompt = f"{current_prompt}\n\n{completion_text}\n\n{feedback}"
                    continue

                # Evaluate on Modal
                try:
                    result = _evaluate_on_modal(code, baseline_orig, baseline_dg)
                    reward = _compute_reward_from_result(result)
                except Exception as e:
                    print(f"  [Turn {turn + 1}] Modal eval failed: {e}")
                    reward = -1.0
                    result = {"compiles": False, "error": str(e)[:200]}

                if reward > best_reward:
                    best_reward = reward

                # Early exit on perfect score
                if reward >= 3.0:
                    break

                # Last turn — no need for feedback
                if turn == max_turns - 1:
                    break

                # Build feedback and extend prompt for next turn
                feedback = _format_feedback(result, reward, turn)
                current_prompt = f"{current_prompt}\n\n{completion_text}\n\n{feedback}"

            all_prompt_ids.append(episode_prompt_ids)
            all_completion_ids.append(episode_completion_ids)
            all_logprobs.append(episode_logprobs)
            all_best_rewards.append(best_reward)

            if (prompt_idx + 1) % 10 == 0 or prompt_idx == 0:
                print(
                    f"  Rollout {prompt_idx + 1}/{len(prompts)}: "
                    f"best_reward={best_reward:.1f}"
                )

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_reward": all_best_rewards,
        }

    return rollout_func


def reward_from_env(completions: list[str], **kwargs: Any) -> list[float]:
    """Extract environment rewards passed via rollout_func kwargs.

    This is the reward_funcs callback for GRPOTrainer. The actual reward
    computation happens in the rollout loop; this just extracts the results.
    """
    env_rewards = kwargs.get("env_reward", [])
    if env_rewards:
        return [float(r) for r in env_rewards]
    return [-1.0] * len(completions)
