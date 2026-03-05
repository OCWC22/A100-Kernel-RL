"""
Stage 2: Rejection Fine-Tuning (RFT) with SFT.

1. Collect trajectories from Stage 1 checkpoint
2. Filter: keep only trajectories with reward >= 1.0 (correct kernels)
3. Train with SFTTrainer on filtered high-quality trajectories

Critical: ByteDance ablation showed skipping RFT causes policy entropy
explosion and training collapse (Table 2, arXiv:2602.24286).
"""
from __future__ import annotations

import os

from trl import SFTConfig, SFTTrainer

from training.model_loader import load_model_and_tokenizer
from training.rft_filter import TrajectoryCollector

STAGE1_OUTPUT = os.getenv("KERNELFORGE_STAGE1_OUTPUT", "outputs/kernelforge-stage1")
OUTPUT_DIR = os.getenv("KERNELFORGE_STAGE2_OUTPUT", "outputs/kernelforge-stage2")
NUM_TRAJECTORIES = int(os.getenv("KERNELFORGE_RFT_TRAJECTORIES", "50"))
MIN_REWARD = float(os.getenv("KERNELFORGE_RFT_MIN_REWARD", "1.0"))


def main():
    """Run Stage 2: collect trajectories, filter, SFT."""
    print("=== Stage 2: Rejection Fine-Tuning ===")

    # Step 1: Collect trajectories using Stage 1 model
    print(f"Collecting {NUM_TRAJECTORIES} trajectories from {STAGE1_OUTPUT}...")
    collector = TrajectoryCollector(model_path=STAGE1_OUTPUT)
    collector.collect_trajectories(num_trajectories=NUM_TRAJECTORIES)

    # Step 2: Filter
    filtered = collector.filter_trajectories(min_reward=MIN_REWARD)
    if not filtered:
        print("No trajectories met quality threshold! Cannot proceed with Stage 2.")
        return

    # Step 3: Save filtered dataset
    os.makedirs("datasets", exist_ok=True)
    rft_dataset = collector.save_rft_dataset(filtered, "datasets/rft_filtered.jsonl")

    # Step 4: Load model from Stage 1 checkpoint and train
    print(f"Loading Stage 1 checkpoint from {STAGE1_OUTPUT}...")
    model, tokenizer = load_model_and_tokenizer(checkpoint_path=STAGE1_OUTPUT)

    config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_train_epochs=3,
        logging_steps=1,
        save_steps=50,
        bf16=True,
        report_to="none",
        max_seq_length=8192,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=rft_dataset,
    )

    print("Starting Stage 2 SFT training on filtered trajectories...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Stage 2 complete. Checkpoint saved to {OUTPUT_DIR}")

    # Print stats
    rewards = [t["reward"] for t in filtered]
    print(f"RFT stats: {len(filtered)} trajectories, "
          f"rewards min={min(rewards):.1f} max={max(rewards):.1f} "
          f"mean={sum(rewards)/len(rewards):.2f}")


if __name__ == "__main__":
    main()
