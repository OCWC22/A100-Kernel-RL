"""
Stage 1: GRPO Warm-up — bootstrap CUDA syntax on easy operators.

Multi-turn agentic training via TRL's rollout_func:
  - 3 turns per episode (model sees errors, iterates)
  - Temperature 1.0 for exploration
  - LR 2e-6 to avoid catastrophic forgetting
  - beta=0.0 (no KL penalty — let model explore freely)
  - G=2 generations, 100 max_steps (hackathon config)
  - vLLM colocate mode for generation

Dataset: CUDA-Agent-Ops-6K easy operators (single-op subset).
"""
import os
os.environ.setdefault('UNSLOTH_VLLM_STANDBY', '1')  # 30%+ memory savings for RL
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from trl import GRPOConfig

from training.custom_grpo_trainer import TRLOOGRPOTrainer
from training.dataset_loader import Dataset, load_training_dataset
from training.model_loader import load_model_and_tokenizer
from training.multi_turn_rollout import make_multi_turn_rollout, reward_from_env
from training.task_support import normalize_task_row

TARGET_GPU = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
OUTPUT_DIR = os.getenv("KERNELFORGE_STAGE1_OUTPUT", "outputs/kernelforge-stage1")
IS_LINUX = sys.platform.startswith("linux")
USE_VLLM = os.getenv("KERNELFORGE_USE_VLLM", "1") == "1" and IS_LINUX
OPTIMIZER = "paged_adamw_8bit" if IS_LINUX else "adamw_torch"
USE_BF16 = IS_LINUX

# Multi-turn configuration
MAX_TURNS = int(os.getenv("KERNELFORGE_STAGE1_MAX_TURNS", "3"))
MAX_STEPS = int(os.getenv("KERNELFORGE_STAGE1_MAX_STEPS", "100"))


# --- Dataset loading ---

def load_stage1_dataset() -> Dataset:
    """Load stage1 prompts from unified dataset loader, with safe fallback."""

    try:
        max_samples = int(os.getenv("CUDA_AGENT_STAGE1_SAMPLES", "512"))
        ds = load_training_dataset(
            stage="stage1",
            ops6k_max=max_samples,
            seed=42,
        )
        if isinstance(ds, Dataset) and len(ds) > 0:
            print(f"Loaded {len(ds)} unified Stage 1 prompts")
            return ds.shuffle(seed=42)
    except Exception as e:
        print(f"Could not load Ops-6K for Stage 1: {e}")

    print("Using fallback Stage 1 prompts with live WCC evaluation support")
    return Dataset.from_list([
        {
            "prompt": (
                f"Write a CUDA Weakly Connected Components kernel for {TARGET_GPU} ({TARGET_ARCH}) "
                "using union-find with path compression."
            ),
            "ops": ["weakly_connected_components"],
            "difficulty": 1,
            "data_source": "fallback_wcc",
        },
        {
            "prompt": (
                f"Write a CUDA Weakly Connected Components kernel for {TARGET_GPU} ({TARGET_ARCH}) "
                "optimized for sparse disconnected graphs with early convergence."
            ),
            "ops": ["weakly_connected_components"],
            "difficulty": 1,
            "data_source": "fallback_wcc",
        },
        {
            "prompt": (
                f"Write a CUDA Weakly Connected Components kernel for {TARGET_GPU} ({TARGET_ARCH}) "
                "using shared memory staging for dense frontiers."
            ),
            "ops": ["weakly_connected_components"],
            "difficulty": 2,
            "data_source": "fallback_wcc",
        },
    ])


# --- Training ---

def main():
    """Run Stage 1 GRPO warm-up with multi-turn agentic loop."""
    print(f"=== Stage 1: Multi-Turn GRPO Warm-up for {TARGET_GPU} ({TARGET_ARCH}) ===")
    print(f"  Max turns per episode: {MAX_TURNS}")
    print(f"  Max training steps: {MAX_STEPS}")

    model, tokenizer = load_model_and_tokenizer()
    dataset = load_stage1_dataset()
    task_rows = [normalize_task_row(row) for row in dataset.to_list()]

    rollout_func = make_multi_turn_rollout(
        max_turns=MAX_TURNS,
        skill_md_gpu=TARGET_GPU.lower(),
        problem_metadata=task_rows,
    )

    config = GRPOConfig(
        learning_rate=2e-6,
        temperature=1.0,         # High exploration
        num_generations=2,
        max_prompt_length=4096,
        max_completion_length=4096,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=MAX_STEPS,
        optim=OPTIMIZER,
        bf16=USE_BF16,
        report_to="none",
        output_dir=OUTPUT_DIR,
        logging_steps=1,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.05,
        use_vllm=USE_VLLM,
        vllm_mode="colocate" if USE_VLLM else "server",
    )

    trainer = TRLOOGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_from_env],
        rollout_func=rollout_func,
        args=config,
        train_dataset=dataset,
    )

    print("Starting Stage 1 training...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Stage 1 complete. Checkpoint saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
