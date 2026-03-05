"""Unified dataset loading for KernelForge training stages."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_cwd = os.getcwd()
_orig_sys_path = list(sys.path)
sys.path = [p for p in sys.path if p not in ("", ".", _cwd)]
try:
    import datasets as _hf_datasets  # noqa: E402
except Exception:  # pragma: no cover - optional local dependency
    _hf_datasets = None
sys.path = _orig_sys_path
Dataset = _hf_datasets.Dataset if _hf_datasets is not None else Any

from training.curriculum import CurriculumManager

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
if str(DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASETS_DIR))

from build_combined_dataset import (  # noqa: E402
    DEFAULT_DG_MANIFEST,
    build_combined_dataset,
    inject_into_curriculum,
)

DEFAULT_COMBINED_PATH = ROOT / "datasets" / "combined_kernelforge.jsonl"
DEFAULT_DG_SFT_PATH = ROOT / "datasets" / "doublegraph_sft.jsonl"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _to_prompt_dataset(rows: list[dict[str, Any]]) -> Dataset:
    if _hf_datasets is None:
        raise RuntimeError("Hugging Face 'datasets' package is required for stage1 prompt Dataset")
    return Dataset.from_list([{"prompt": str(r.get("prompt", ""))} for r in rows if r.get("prompt")])


def load_training_dataset(
    stage: str,
    dg_manifest: str = str(DEFAULT_DG_MANIFEST),
    ops6k_max: int | None = 1024,
    seed: int = 42,
    combined_output: str = str(DEFAULT_COMBINED_PATH),
    sft_path: str = str(DEFAULT_DG_SFT_PATH),
    curriculum_manager: CurriculumManager | None = None,
) -> Dataset | list[dict[str, Any]]:
    """Load dataset appropriate for each training stage.

    stage1:
        Prompt Dataset with easy Ops-6K samples + doubleGraph base kernels.
    stage2:
        SFT rows from datasets/doublegraph_sft.jsonl.
    stage3:
        Full combined rows; optionally injected into provided CurriculumManager.
    """
    stage_key = stage.strip().lower()

    if stage_key not in {"stage1", "stage2", "stage3"}:
        raise ValueError("stage must be one of: stage1, stage2, stage3")

    if stage_key == "stage2":
        sft_file = Path(sft_path)
        if not sft_file.exists():
            return []
        return _read_jsonl(sft_file)

    combined_path = Path(combined_output)
    if combined_path.exists():
        rows = _read_jsonl(combined_path)
    else:
        rows = build_combined_dataset(
            dg_path=dg_manifest,
            ops6k_max=ops6k_max,
            seed=seed,
        )

    if stage_key == "stage1":
        stage1_rows = [
            row
            for row in rows
            if int(row.get("difficulty", 1)) == 1
            or (row.get("data_source") == "doublegraph_a100" and int(row.get("difficulty", 1)) == 2)
        ]
        return _to_prompt_dataset(stage1_rows)

    if curriculum_manager is not None:
        inject_into_curriculum(curriculum_manager, rows)
    return rows
