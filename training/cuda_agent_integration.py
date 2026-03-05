"""Utilities to integrate CUDA-Agent-Ops-6K into KernelForge training prompts."""

from __future__ import annotations

import ast
import json
import os
from typing import Any

from datasets import Dataset, load_dataset

CUDA_AGENT_DATASET_ID = "BytedTsinghua-SIA/CUDA-Agent-Ops-6K"
TARGET_GPU_NAME = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")


def _parse_ops(ops: Any) -> list[str]:
    """Best-effort parser for the `ops` field."""
    if isinstance(ops, list):
        return [str(x) for x in ops]

    if isinstance(ops, str):
        text = ops.strip()
        if not text:
            return []

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                continue

        return [text]

    return [str(ops)] if ops is not None else []


def _build_cuda_prompt(example: dict[str, Any], max_code_chars: int = 6000) -> str | None:
    """Convert one CUDA-Agent sample into a GRPO-compatible prompt."""
    code = str(example.get("code", "")).strip()
    if not code:
        return None

    ops = _parse_ops(example.get("ops"))
    ops_desc = ", ".join(ops) if ops else "unknown operator pipeline"
    data_source = str(example.get("data_source", "unknown"))

    if len(code) > max_code_chars:
        code = code[:max_code_chars] + "\n# [truncated]"

    return (
        f"Write a high-performance CUDA kernel implementation for NVIDIA {TARGET_GPU_NAME} ({TARGET_ARCH}) "
        "that matches the semantics of the PyTorch reference below.\n\n"
        f"Reference ops: {ops_desc}\n"
        f"Data source: {data_source}\n\n"
        "Reference implementation:\n"
        "```python\n"
        f"{code}\n"
        "```\n\n"
        "Return CUDA/C++ code only, with a callable binding function and correctness-focused logic."
    )


def load_cuda_agent_prompt_dataset(max_samples: int | None = 1024, seed: int = 42) -> Dataset:
    """Load CUDA-Agent-Ops-6K and convert rows to prompt-only dataset."""
    ds = load_dataset(CUDA_AGENT_DATASET_ID, split="train")

    if max_samples is not None:
        max_samples = max(0, int(max_samples))
        if max_samples == 0:
            return Dataset.from_list([])
        if len(ds) > max_samples:
            ds = ds.shuffle(seed=seed).select(range(max_samples))

    prompts: list[dict[str, Any]] = []
    for row in ds:
        prompt = _build_cuda_prompt(row)
        if prompt:
            prompts.append({
                "prompt": prompt,
                "task_code": str(row.get("code", "")),
                "ops": str(row.get("ops", "")),
                "data_source": str(row.get("data_source", "")),
            })

    return Dataset.from_list(prompts)


def load_cuda_agent_prompt_texts(max_samples: int = 128, seed: int = 42) -> list[str]:
    """Return CUDA-Agent prompts as a list of strings (for RFT prompt pools)."""
    ds = load_cuda_agent_prompt_dataset(max_samples=max_samples, seed=seed)
    return [str(row["prompt"]) for row in ds if row.get("prompt")]
