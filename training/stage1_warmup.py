"""
Stage 1: GRPO Warm-up — bootstrap CUDA syntax on easy operators.

Multi-turn agentic training via TRL's rollout_func:
  - 3 turns per episode (model sees errors, iterates)
  - Higher temperature (0.9) for exploration
  - Lower LR (3e-6) to avoid catastrophic forgetting
  - beta=0.0 (no KL penalty — let model explore freely)
  - 300 max_steps (was 100; increased for multi-turn budget)
  - vLLM colocate mode for generation

Dataset: CUDA-Agent-Ops-6K easy operators (single-op subset).
See docs/TRAINING_PLAN.md for full rationale.
"""
from __future__ import annotations

import os

from datasets import Dataset
from trl import GRPOConfig

from training.custom_grpo_trainer import TRLOOGRPOTrainer
from training.dataset_loader import load_training_dataset
from training.model_loader import load_model_and_tokenizer
from training.multi_turn_rollout import make_multi_turn_rollout, reward_from_env

TARGET_GPU = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
OUTPUT_DIR = os.getenv("KERNELFORGE_STAGE1_OUTPUT", "outputs/kernelforge-stage1")

# Multi-turn configuration
MAX_TURNS = int(os.getenv("KERNELFORGE_STAGE1_MAX_TURNS", "3"))
MAX_STEPS = int(os.getenv("KERNELFORGE_STAGE1_MAX_STEPS", "300"))


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

    print("Using fallback Stage 1 prompts")
    return Dataset.from_list([
        {"prompt": f"Write a CUDA vector addition kernel for {TARGET_GPU} ({TARGET_ARCH}). Take float* A, float* B, float* C, int N."},
        {"prompt": f"Write a CUDA ReLU activation kernel for {TARGET_GPU} ({TARGET_ARCH}). Apply max(0, x) to a float array."},
        {"prompt": f"Write a CUDA softmax kernel for {TARGET_GPU} ({TARGET_ARCH}). Compute row-wise softmax for a 2D matrix."},
        {"prompt": f"Write a CUDA matrix multiplication kernel using shared memory tiling for {TARGET_GPU} ({TARGET_ARCH})."},
        {"prompt": f"Write a CUDA GELU activation kernel for {TARGET_GPU} ({TARGET_ARCH})."},
        {"prompt": f"Write a CUDA layer normalization kernel for {TARGET_GPU} ({TARGET_ARCH})."},
        {"prompt": f"Write a CUDA batch normalization kernel for {TARGET_GPU} ({TARGET_ARCH})."},
        {"prompt": f"Write a CUDA element-wise sigmoid kernel for {TARGET_GPU} ({TARGET_ARCH})."},
    ])


# --- Training ---

def main():
    """Run Stage 1 GRPO warm-up with multi-turn agentic loop."""
    print(f"=== Stage 1: Multi-Turn GRPO Warm-up for {TARGET_GPU} ({TARGET_ARCH}) ===")
    print(f"  Max turns per episode: {MAX_TURNS}")
    print(f"  Max training steps: {MAX_STEPS}")

    model, tokenizer = load_model_and_tokenizer()
    dataset = load_stage1_dataset()

    rollout_func = make_multi_turn_rollout(
        max_turns=MAX_TURNS,
        skill_md_gpu=TARGET_GPU.lower(),
    )

    config = GRPOConfig(
        learning_rate=3e-6,
        temperature=0.9,         # High exploration
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=4096,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=MAX_STEPS,
        optim="paged_adamw_8bit",
        bf16=True,
        report_to="none",
        output_dir=OUTPUT_DIR,
        logging_steps=1,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.05,
        use_vllm=True,
        vllm_mode="colocate",
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
