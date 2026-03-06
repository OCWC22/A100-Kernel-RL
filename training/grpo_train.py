"""CLI entrypoint for the KernelForge training pipeline."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from training.dataset_loader import load_training_dataset
from training.task_support import summarize_tasks

REQUIRED_PACKAGES = ("trl", "transformers", "torch", "modal")
if sys.platform.startswith("linux"):
    REQUIRED_PACKAGES += ("peft",)


def _check_dependencies(packages: Iterable[str] = REQUIRED_PACKAGES) -> list[str]:
    """Return a list of missing Python packages."""
    missing = []
    for package in packages:
        try:
            importlib.metadata.version(package)
        except Exception:
            missing.append(package)
    return missing


def _dataset_summary() -> dict[str, dict[str, int]]:
    """Summarize the supported training dataset."""
    rows = load_training_dataset(stage="stage3", ops6k_max=int(os.getenv("CUDA_AGENT_STAGE1_SAMPLES", "128")))
    return summarize_tasks(rows)


def preflight() -> None:
    """Run non-destructive checks before training."""
    missing = _check_dependencies()
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "Missing required training dependencies: "
            f"{missing_str}. Install them with `uv sync --extra train --extra modal --extra openenv`."
        )

    try:
        import modal
        if not getattr(modal.config, "token_id", None):
            print("WARNING: Modal authentication not configured. "
                  "Run `modal token new` or set MODAL_TOKEN_ID/MODAL_TOKEN_SECRET env vars. "
                  "Training will fail at evaluation time without Modal auth.")
    except ImportError:
        print("WARNING: modal package not installed. Eval dispatch will fail.")

    summary = _dataset_summary()
    print("Dataset summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def main() -> None:
    """Run a specific training stage or the full pipeline."""
    parser = argparse.ArgumentParser(description="KernelForge training launcher")
    parser.add_argument(
        "--stage",
        choices=("stage1", "stage2", "stage3", "pipeline"),
        default=os.getenv("KERNELFORGE_STAGE", "stage1"),
        help="Training stage to run",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate environment and dataset wiring without starting training",
    )
    args = parser.parse_args()

    preflight()
    if args.preflight_only:
        print("Preflight checks passed.")
        return

    if args.stage in {"stage1", "pipeline"}:
        from training.stage1_warmup import main as stage1_main

        stage1_main()

    if args.stage in {"stage2", "pipeline"}:
        from training.stage2_rft import main as stage2_main

        stage2_main()

    if args.stage in {"stage3", "pipeline"}:
        from training.stage3_grpo import main as stage3_main

        stage3_main()


if __name__ == "__main__":
    main()
