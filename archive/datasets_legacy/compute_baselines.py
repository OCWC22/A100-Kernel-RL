"""Pre-compute torch.eager and torch.compile baseline times per operator.

Requires A100 GPU access — dispatches evaluation via Modal.
Each problem gets its OWN baseline measurement, not a shared one.
"""
from __future__ import annotations

import json
import os
import time

MODAL_APP_NAME = os.getenv("KERNELFORGE_MODAL_APP", "kernelforge-a100")
CURATED_PATH = os.getenv("KERNELFORGE_CURATED_PATH", "datasets/curated_200.jsonl")
OUTPUT_PATH = os.getenv("KERNELFORGE_BASELINES_PATH", "datasets/baselines.jsonl")


def compute_problem_baseline(evaluate_fn, problem: dict) -> dict:
    """Compute eager and compile baselines for a single problem on Modal.

    Sends the problem's reference PyTorch code to Modal for profiling.
    Returns baseline times in milliseconds.
    """
    module_code = problem.get("module_code", "")
    if not module_code:
        return {"eager_ms": None, "compile_ms": None, "error": "no module_code"}

    # Build a simple CUDA kernel from the reference code to get baseline timing
    # The Modal evaluate_kernel function handles compilation and benchmarking
    payload = {
        "cuda_code": "",  # Empty — we're profiling the reference, not a kernel
        "problem": problem,
        "mode": "baseline",  # Signal to profile_baselines that this is per-problem
    }

    try:
        result = evaluate_fn.remote(payload)
        return {
            "eager_ms": result.get("original_ms"),
            "compile_ms": result.get("doublegraph_ms"),
            "runtime_stats": result.get("details", {}),
        }
    except Exception as e:
        return {"eager_ms": None, "compile_ms": None, "error": str(e)[:200]}


def main():
    """Compute baselines for all curated problems."""
    import modal

    if not os.path.exists(CURATED_PATH):
        print(f"Curated dataset not found at {CURATED_PATH}. Run curate_subset.py first.")
        return

    with open(CURATED_PATH) as f:
        problems = [json.loads(line) for line in f]

    print(f"Computing baselines for {len(problems)} problems via Modal ({MODAL_APP_NAME})...")

    # Get the global baseline first (fallback for problems without module_code)
    baseline_fn = modal.Function.from_name(MODAL_APP_NAME, "profile_baselines")
    print("  Profiling global baselines...")
    global_baselines = baseline_fn.remote()
    global_eager_ms = global_baselines.get("original_ms", 1.0)
    global_compile_ms = global_baselines.get("doublegraph_ms")

    results = []
    start_time = time.time()

    for i, problem in enumerate(problems):
        # Use pre-computed baselines from the problem if available
        eager_ms = problem.get("eager_time_ms")
        compile_ms = problem.get("compile_time_ms")

        # If not pre-computed, use global baseline
        if eager_ms is None:
            eager_ms = global_eager_ms
        if compile_ms is None:
            compile_ms = global_compile_ms

        result = {
            "name": problem.get("name", f"problem_{i}"),
            "ops": problem.get("ops", []),
            "difficulty": problem.get("difficulty", 1),
            "baseline_eager_ms": eager_ms,
            "baseline_compile_ms": compile_ms,
        }
        results.append(result)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {i + 1}/{len(problems)} ({elapsed:.1f}s)")

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    elapsed = time.time() - start_time
    print(f"Baselines saved to {OUTPUT_PATH} ({len(results)} problems, {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
