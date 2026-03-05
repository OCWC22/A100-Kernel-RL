"""
Stage 3: GRPO with Curriculum — P3 optional demo (10 steps, local-only eval).

Multi-turn agentic training via TRL's rollout_func:
  - 3 turns per episode (reduced from 5 — TRLOO bias compounds with turns)
  - CurriculumManager for progressive difficulty
  - Temperature 0.7 for exploitation
  - LR 5e-6 for faster convergence
  - 10 max_steps (P3 demo — only if Gate G-0.8 passes)
  - G=2 (reduced from 4 — fewer zero-gradient steps)
  - Local nvcc eval only, no Modal
  - vLLM colocate mode for generation

GRPO is experimental. SkyDiscover + SFT are primary hedges.
See docs/GRPO_DEEP_DIVE.md GRPO-14 for full stacked architecture.
"""
from __future__ import annotations

import os

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from training.model_loader import load_model_and_tokenizer
from training.curriculum import CurriculumManager
from training.multi_turn_rollout import make_multi_turn_rollout

TARGET_GPU = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
MODAL_APP_NAME = os.getenv("KERNELFORGE_MODAL_APP", "kernelforge-a100")
STAGE2_OUTPUT = os.getenv("KERNELFORGE_STAGE2_OUTPUT", "outputs/kernelforge-stage2")
OUTPUT_DIR = os.getenv("KERNELFORGE_STAGE3_OUTPUT", "outputs/kernelforge-stage3")

# Multi-turn configuration (P3 demo defaults)
MAX_TURNS = int(os.getenv("KERNELFORGE_STAGE3_MAX_TURNS", "3"))
MAX_STEPS = int(os.getenv("KERNELFORGE_STAGE3_MAX_STEPS", "10"))
LOCAL_ONLY = os.getenv("KERNELFORGE_STAGE3_LOCAL_ONLY", "1") == "1"


# Global curriculum manager — shared between reward function and training loop
curriculum = CurriculumManager()


def reward_from_env_with_curriculum(completions, **kwargs) -> list[float]:
    """Extract environment rewards and feed them to the curriculum manager."""
    env_rewards = kwargs.get("env_reward", [])
    if not env_rewards:
        return [-1.0] * len(completions)

    rewards = [float(r) for r in env_rewards]

    # Feed each reward to curriculum for promotion/demotion
    for r in rewards:
        transition = curriculum.record_reward(r)
        if transition:
            print(f"  Curriculum transition: {transition} -> now in phase '{curriculum.phase_name}'")

    return rewards


# --- Dataset: curriculum-aware prompt generation ---

def build_curriculum_dataset(num_prompts: int = 200) -> Dataset:
    """Generate prompts from curriculum manager, sampling from current phase."""
    prompts = []
    for _ in range(num_prompts):
        problem = curriculum.get_problem()
        prompts.append({"prompt": problem["prompt"]})
    return Dataset.from_list(prompts)


# --- Training ---

def main():
    """Run Stage 3 GRPO with curriculum and multi-turn agentic loop."""
    print(f"=== Stage 3: Multi-Turn GRPO with Curriculum for {TARGET_GPU} ({TARGET_ARCH}) ===")
    print(f"  Starting phase: {curriculum.phase_name}")
    print(f"  Max turns per episode: {MAX_TURNS}")
    print(f"  Max training steps: {MAX_STEPS}")

    model, tokenizer = load_model_and_tokenizer(checkpoint_path=STAGE2_OUTPUT)
    dataset = build_curriculum_dataset(num_prompts=200)

    rollout_func = make_multi_turn_rollout(
        max_turns=MAX_TURNS,
        skill_md_gpu=TARGET_GPU.lower(),
    )

    config = GRPOConfig(
        learning_rate=5e-6,
        temperature=0.7,         # Lower temp for exploitation
        num_generations=2,          # Reduced from 4 (fewer zero-gradient steps)
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

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_from_env_with_curriculum],
        rollout_func=rollout_func,
        args=config,
        train_dataset=dataset,
    )

    print("Starting Stage 3 training...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Print curriculum summary
    status = curriculum.status()
    print(f"\nStage 3 complete. Checkpoint saved to {OUTPUT_DIR}")
    print(f"Final curriculum phase: {status['phase']} ({status['phase_idx']+1}/{status['total_phases']})")
    print(f"Phase transitions: {status['transitions']}")
    print(f"Hit rate: {status['hit_rate']:.1%}, Positive rate: {status['positive_rate']:.1%}")


if __name__ == "__main__":
    main()
