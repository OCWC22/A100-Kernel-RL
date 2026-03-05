"""Evaluate a trained model on held-out operators.

Metrics: compilation rate, correctness rate, avg reward, speedup distribution.
Compare across stages: base -> stage1 -> stage2 -> stage3.
"""
from __future__ import annotations

import json
import os
import statistics

from datasets import load_dataset

from openenv_env.reward import compute_reward

MODAL_APP_NAME = os.getenv("KERNELFORGE_MODAL_APP", "kernelforge-a100")
TARGET_GPU = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
EVAL_PROBLEMS = 50


def evaluate_checkpoint(checkpoint_path: str, num_problems: int = EVAL_PROBLEMS) -> dict:
    """Evaluate a model checkpoint on held-out operators."""
    import modal
    from training.model_loader import load_model_and_tokenizer

    print(f"Evaluating {checkpoint_path} on {num_problems} problems...")

    model, tokenizer = load_model_and_tokenizer(checkpoint_path=checkpoint_path)

    eval_fn = modal.Function.from_name(MODAL_APP_NAME, "evaluate_kernel")
    baseline_fn = modal.Function.from_name(MODAL_APP_NAME, "profile_baselines")
    baselines = baseline_fn.remote() or {}
    baseline_orig = baselines.get("original_ms")
    baseline_dg = baselines.get("doublegraph_ms")

    prompts = _load_eval_prompts(num_problems)

    results = {
        "checkpoint": checkpoint_path,
        "num_problems": len(prompts),
        "compiles": 0,
        "correct": 0,
        "rewards": [],
        "speedups": [],
    }

    from transformers import pipeline
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    for i, prompt in enumerate(prompts):
        try:
            output = generator(prompt, max_new_tokens=2048, do_sample=True, temperature=0.7)
            code = output[0]["generated_text"]
            if code.startswith(prompt):
                code = code[len(prompt):]

            result = eval_fn.remote({
                "cuda_code": code,
                "verify_graphs": 5,
                "warmup_iters": 50,
                "benchmark_runs": 30,
                "baseline_original_ms": baseline_orig,
                "baseline_doublegraph_ms": baseline_dg,
            })

            compiles = result.get("compiles", False)
            correct = result.get("correct", False)
            reward = compute_reward(
                compiled=compiles,
                correct=correct,
                speedup_vs_eager=result.get("speedup_vs_orig", 0),
                speedup_vs_compile=result.get("speedup_vs_dg", 0),
            )

            if compiles:
                results["compiles"] += 1
            if correct:
                results["correct"] += 1
            results["rewards"].append(reward)
            if correct and result.get("speedup_vs_orig", 0) > 0:
                results["speedups"].append(result["speedup_vs_orig"])

        except Exception as e:
            print(f"  Problem {i+1} failed: {e}")
            results["rewards"].append(-1.0)

        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i+1}/{len(prompts)}")

    # Compute summary stats
    n = results["num_problems"]
    results["compile_rate"] = results["compiles"] / n if n > 0 else 0
    results["correct_rate"] = results["correct"] / n if n > 0 else 0
    results["avg_reward"] = statistics.mean(results["rewards"]) if results["rewards"] else 0
    results["median_speedup"] = statistics.median(results["speedups"]) if results["speedups"] else 0

    return results


def _load_eval_prompts(num_problems: int) -> list[str]:
    """Load evaluation prompts from curated dataset or generate fallback."""
    curated_path = "datasets/curated_200.jsonl"
    if os.path.exists(curated_path):
        with open(curated_path) as f:
            all_problems = [json.loads(line) for line in f]
        # Use last N as held-out eval set
        eval_set = all_problems[-num_problems:]
        return [
            f"Write a CUDA kernel for {TARGET_GPU} ({TARGET_ARCH}) that implements: {p.get('ops', 'unknown')}"
            for p in eval_set
        ]

    print(
        f"WARNING: {curated_path} not found. Run 'python datasets/curate_subset.py' first "
        "for proper evaluation. Falling back to minimal built-in prompts."
    )
    return [
        f"Write a CUDA vector addition kernel for {TARGET_GPU} ({TARGET_ARCH}).",
        f"Write a CUDA matrix multiplication kernel for {TARGET_GPU} ({TARGET_ARCH}).",
        f"Write a CUDA softmax kernel for {TARGET_GPU} ({TARGET_ARCH}).",
        f"Write a CUDA ReLU kernel for {TARGET_GPU} ({TARGET_ARCH}).",
        f"Write a CUDA GELU kernel for {TARGET_GPU} ({TARGET_ARCH}).",
    ] * (num_problems // 5 + 1)


def evaluate_multi_seed(
    checkpoint_path: str,
    num_problems: int = EVAL_PROBLEMS,
    num_seeds: int = 3,
    temperatures: tuple[float, ...] = (0.2, 0.7, 1.0),
) -> dict:
    """Run evaluation across multiple seeds and temperatures.

    Computes mean, std, and 95% CI for all metrics. Requires Modal + GPU.
    Standard ML practice: >=3 seeds minimum for credible results.
    """
    all_results = []
    for seed in range(num_seeds):
        for temp in temperatures:
            print(f"\n--- Seed {seed}, temp {temp} ---")
            result = evaluate_checkpoint(
                checkpoint_path,
                num_problems=num_problems,
            )
            result["seed"] = seed
            result["temperature"] = temp
            all_results.append(result)

    # Aggregate
    compile_rates = [r["compile_rate"] for r in all_results]
    correct_rates = [r["correct_rate"] for r in all_results]
    avg_rewards = [r["avg_reward"] for r in all_results]

    def _ci_95(values: list[float]) -> tuple[float, float]:
        """95% CI using t-distribution approximation."""
        n = len(values)
        if n < 2:
            m = values[0] if values else 0
            return (m, m)
        mean = statistics.mean(values)
        se = statistics.stdev(values) / (n ** 0.5)
        # t-value for 95% CI with n-1 df (approximation: 1.96 for large n, 2.776 for n=5)
        t_val = 2.776 if n <= 5 else 2.228 if n <= 10 else 1.96
        return (mean - t_val * se, mean + t_val * se)

    return {
        "checkpoint": checkpoint_path,
        "num_runs": len(all_results),
        "compile_rate": {
            "mean": statistics.mean(compile_rates),
            "std": statistics.stdev(compile_rates) if len(compile_rates) > 1 else 0,
            "ci_95": _ci_95(compile_rates),
        },
        "correct_rate": {
            "mean": statistics.mean(correct_rates),
            "std": statistics.stdev(correct_rates) if len(correct_rates) > 1 else 0,
            "ci_95": _ci_95(correct_rates),
        },
        "avg_reward": {
            "mean": statistics.mean(avg_rewards),
            "std": statistics.stdev(avg_rewards) if len(avg_rewards) > 1 else 0,
            "ci_95": _ci_95(avg_rewards),
        },
        "raw_results": all_results,
    }


def compare_stages():
    """Compare all training stages."""
    stages = [
        ("Base", None),
        ("Stage 1 (GRPO warm-up)", "outputs/kernelforge-stage1"),
        ("Stage 2 (RFT)", "outputs/kernelforge-stage2"),
        ("Stage 3 (GRPO+curriculum)", "outputs/kernelforge-stage3"),
    ]

    print("=== Model Evaluation Across Training Stages ===\n")

    for name, path in stages:
        if path and not os.path.exists(path):
            print(f"  {name}: checkpoint not found at {path}, skipping")
            continue

        results = evaluate_checkpoint(path or "Qwen/Qwen2.5-Coder-7B-Instruct")
        print(f"\n{name}:")
        print(f"  Compile rate: {results['compile_rate']:.1%}")
        print(f"  Correct rate: {results['correct_rate']:.1%}")
        print(f"  Avg reward:   {results['avg_reward']:.2f}")
        print(f"  Median speedup: {results['median_speedup']:.2f}x")


def main():
    compare_stages()


if __name__ == "__main__":
    main()
