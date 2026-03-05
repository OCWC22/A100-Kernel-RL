"""Download CUDA-Agent-Ops-6K dataset from HuggingFace."""
from __future__ import annotations

import importlib
import json
import os
import sys

# Avoid shadowing: local datasets/ dir conflicts with HuggingFace 'datasets' package.
# Remove CWD from path temporarily to import the real package.
_cwd = os.getcwd()
sys.path = [p for p in sys.path if p not in ("", ".", _cwd)]
import datasets as _hf_datasets  # noqa: E402
sys.path.insert(0, "")  # restore
load_dataset = _hf_datasets.load_dataset


DATASET_ID = "BytedTsinghua-SIA/CUDA-Agent-Ops-6K"
OUTPUT_DIR = os.getenv("KERNELFORGE_DATA_DIR", "datasets")


def main():
    """Download and save Ops-6K as local JSONL."""
    print(f"Downloading {DATASET_ID}...")
    ds = load_dataset(DATASET_ID, split="train")
    print(f"Downloaded {len(ds)} examples")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "ops6k_full.jsonl")

    with open(output_path, "w") as f:
        for row in ds:
            f.write(json.dumps(dict(row)) + "\n")

    print(f"Saved to {output_path}")

    # Print stats
    ops_counts = {}
    for row in ds:
        ops = row.get("ops", "unknown")
        if isinstance(ops, list):
            key = len(ops)
        else:
            key = 1
        ops_counts[key] = ops_counts.get(key, 0) + 1

    print("\nOperator count distribution:")
    for k in sorted(ops_counts):
        print(f"  {k}-op: {ops_counts[k]} ({ops_counts[k]/len(ds)*100:.1f}%)")


if __name__ == "__main__":
    main()
