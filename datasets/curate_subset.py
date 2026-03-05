"""Curate a 200-problem subset from Ops-6K for curriculum training."""
from __future__ import annotations

import json
import os
import random
import sys

# Avoid shadowing: local datasets/ dir conflicts with HuggingFace 'datasets' package.
_cwd = os.getcwd()
sys.path = [p for p in sys.path if p not in ("", ".", _cwd)]
import datasets as _hf_datasets  # noqa: E402
sys.path.insert(0, "")
load_dataset = _hf_datasets.load_dataset


DATASET_ID = "BytedTsinghua-SIA/CUDA-Agent-Ops-6K"
OUTPUT_PATH = os.getenv("KERNELFORGE_CURATED_PATH", "datasets/curated_200.jsonl")
SEED = 42


def count_ops(row: dict) -> int:
    """Count number of operators in a row."""
    ops = row.get("ops", [])
    if isinstance(ops, list):
        return len(ops)
    if isinstance(ops, str):
        try:
            import ast
            parsed = ast.literal_eval(ops)
            if isinstance(parsed, list):
                return len(parsed)
        except Exception:
            pass
        return 1
    return 1


def main():
    """Select 200 problems: 50 easy, 75 medium, 75 hard."""
    print(f"Loading {DATASET_ID}...")
    ds = load_dataset(DATASET_ID, split="train")
    print(f"Total: {len(ds)} examples")

    random.seed(SEED)

    easy = []    # 1 op
    medium = []  # 2 ops
    hard = []    # 3+ ops

    for row in ds:
        n = count_ops(row)
        if n == 1:
            easy.append(dict(row))
        elif n == 2:
            medium.append(dict(row))
        else:
            hard.append(dict(row))

    print(f"Distribution: {len(easy)} easy, {len(medium)} medium, {len(hard)} hard")

    random.shuffle(easy)
    random.shuffle(medium)
    random.shuffle(hard)

    selected = (
        easy[:50] +
        medium[:75] +
        hard[:75]
    )

    # Tag with difficulty
    for item in selected[:50]:
        item["difficulty"] = 1
    for item in selected[50:125]:
        item["difficulty"] = 2
    for item in selected[125:]:
        item["difficulty"] = 3

    random.shuffle(selected)

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for item in selected:
            f.write(json.dumps(item) + "\n")

    print(f"Curated {len(selected)} problems -> {OUTPUT_PATH}")
    print(f"  Easy (1-op): {min(50, len(easy))}")
    print(f"  Medium (2-op): {min(75, len(medium))}")
    print(f"  Hard (3+ op): {min(75, len(hard))}")


if __name__ == "__main__":
    main()
