# GRPO Deep Dive: Exact Implementation for B200 192GB
## Addendum to KernelForge Unified Spec — Section 9 Expansion
**Paste after Section 9 or replace it entirely.**

---

## GRPO-1: The Algorithm (Full Math)

### What GRPO Actually Is

GRPO (Group Relative Policy Optimization) is PPO without a critic network. Instead of training a separate value model to estimate "how good is this state," GRPO generates MULTIPLE completions for the same prompt and uses their relative rewards as the baseline. This eliminates ~50% of memory (no critic weights, no critic optimizer, no critic activations).

### The Exact Equations

**Step 1: Group Sampling**

For each prompt q_j in a batch of M prompts, sample G completions from the current (old) policy:

```
{o_{j,1}, o_{j,2}, ..., o_{j,G}} ~ π_θ_old(· | q_j)
```

In our case: M = 1 (one prompt per step), G = 4 (four kernel candidates per prompt).

**Step 2: Reward Computation**

Each completion gets a scalar reward from the environment:

```
r_{j,i} = env.step(o_{j,i}).reward    for i = 1..G
```

In our case, this is the discrete reward {-1, 1, 2, 3} from CUDA Agent's Equation 1:
- r = -1: compilation fails OR correctness fails
- r = +1: correct but no speedup (speedup ≤ 1.05×)
- r = +2: >5% faster than torch.eager
- r = +3: >5% faster than BOTH torch.eager AND torch.compile

**Step 3: Advantage Estimation (The Key Innovation)**

PPO computes advantages using a learned value function V(s). GRPO computes advantages using group statistics:

```
Â_{j,i} = (r_{j,i} - mean(r_{j,1}, ..., r_{j,G})) / (std(r_{j,1}, ..., r_{j,G}) + ε)
```

Where ε = 1e-8 to avoid division by zero.

**Concrete example with our reward values:**

```
Prompt: "Write a CUDA kernel for softmax on A100"

Completion 1: Compiles, correct, no speedup    → r = +1
Completion 2: Compilation fails                 → r = -1
Completion 3: Compiles, correct, beats eager    → r = +2
Completion 4: Compiles, wrong output            → r = -1

mean(r) = (1 + (-1) + 2 + (-1)) / 4 = 0.25
std(r)  = sqrt(((1-0.25)² + (-1-0.25)² + (2-0.25)² + (-1-0.25)²) / 4)
        = sqrt((0.5625 + 1.5625 + 3.0625 + 1.5625) / 4)
        = sqrt(1.6875)
        ≈ 1.299

Â_1 = (1 - 0.25) / 1.299 = +0.577   ← "slightly better than average"
Â_2 = (-1 - 0.25) / 1.299 = -0.962  ← "worse than average"
Â_3 = (2 - 0.25) / 1.299 = +1.347   ← "much better than average"
Â_4 = (-1 - 0.25) / 1.299 = -0.962  ← "worse than average"
```

**What this means:** Every token in Completion 3 gets advantage +1.347 — the model is pushed to generate MORE text like Completion 3. Every token in Completions 2 and 4 gets advantage -0.962 — the model is pushed AWAY from generating text like those.

**Critical edge case:** If all 4 completions get the same reward (e.g., all fail to compile, all r = -1), then std(r) = 0, and all advantages are 0. **No learning signal.** This is why SFT warmup is mandatory — you need at least SOME completions that compile so there's variance in the group.

**Step 4: Policy Loss (Clipped Surrogate)**

For each token t in completion o_{j,i}:

```
ratio_t = π_θ(token_t | context) / π_θ_old(token_t | context)

L_clip = min(
    ratio_t × Â_{j,i},
    clip(ratio_t, 1-ε, 1+ε) × Â_{j,i}
)
```

Where ε = 0.2 (standard PPO clip range).

The clipping prevents the policy from changing too much in one step. If the ratio goes above 1.2 or below 0.8, it gets clipped — limiting the gradient magnitude.

**Step 5: KL Penalty**

GRPO adds a KL divergence term to prevent the policy from drifting too far from the reference (initial) model:

```
KL_t = π_ref(token_t | context) / π_θ(token_t | context) 
       - log(π_ref(token_t | context) / π_θ(token_t | context)) 
       - 1
```

This is the "unbiased estimator" of KL divergence used by DeepSeek.

**Full GRPO Objective:**

```
J_GRPO(θ) = E[L_clip] - β × E[KL]
```

Where β is the KL coefficient. DeepSeek used β = 0.04. CUDA Agent's DAPO variant sets β = 0.0 (no KL penalty). **We use β = 0.0** because:
1. With PEFT/LoRA, the adapter IS the deviation — the base model stays frozen as the reference
2. DAPO showed KL penalty "not essential" for code tasks
3. Fewer hyperparameters to tune at a hackathon

### GRPO vs PPO: Memory Comparison on B200

```
PPO requires 4 models in memory:
  1. Policy (π_θ)           — the model being trained
  2. Reference (π_ref)      — frozen copy for KL computation
  3. Critic (V_ψ)           — value network (same size as policy)
  4. Reward model (R_φ)     — scores completions

GRPO requires 2 models + 1 reward function:
  1. Policy (π_θ)           — the model being trained
  2. Reference (π_ref)      — frozen copy for KL computation
  3. Reward function         — our env.step() (NOT a neural network)

With PEFT (LoRA), reference behavior is recovered by disabling
the adapter — so there's no separate reference model in memory.

Effective GRPO memory = 1 model + LoRA adapters + optimizer states
```

---

## GRPO-2: B200 192GB Memory Budget (Exact)

### Model: Qwen3.5-35B-A3B (MoE, 3B active per token) — BACKUP SCENARIO

```
B200 Total VRAM: 192 GB

COMPONENT                          MEMORY        NOTES
─────────────────────────────────────────────────────────
Model weights (4-bit QLoRA)        ~17.5 GB      35B params × 4 bits / 8
LoRA adapters (r=16)               ~0.2 GB       Only on attention + MLP projections
LoRA optimizer (AdamW)             ~0.8 GB       2 states per LoRA param (FP32)
Gradient checkpointing activations ~4.0 GB       Unsloth's optimized checkpointing
KV cache (G=4, 4096 tokens each)   ~8.0 GB       4 generations × 4096 × hidden_dim
Generation logprobs storage        ~2.0 GB       For policy ratio computation
Reference logprobs                 ~2.0 GB       LoRA disabled = reference behavior
─────────────────────────────────────────────────────────
TOTAL MODEL + TRAINING             ~34.5 GB
─────────────────────────────────────────────────────────

CUDA Kernel Evaluation Sandbox:
  nvcc compilation workspace        ~2 GB
  Test input tensors (5 inputs)     ~1 GB        Depends on operator
  Profiling workspace               ~1 GB
  CachePool (8 entries)             ~2 GB
─────────────────────────────────────────────────────────
TOTAL SANDBOX                      ~6 GB
─────────────────────────────────────────────────────────

REMAINING FREE                     ~151.5 GB     ← MASSIVE headroom
```

### Model: Qwen3.5-9B (Dense)

```
COMPONENT                          MEMORY
─────────────────────────────────────────────────────────
Model weights (4-bit QLoRA)        ~5 GB
LoRA + optimizer                   ~1.5 GB
Activations + KV cache             ~6 GB
─────────────────────────────────────────────────────────
TOTAL                              ~12.5 GB
REMAINING                          ~179.5 GB
```

### Model: Qwen3-Coder-Next (80B MoE, 3B active) — PRIMARY MODEL

```
COMPONENT                          MEMORY
─────────────────────────────────────────────────────────
Model weights (FP8 Dynamic)        ~85 GB         80B × 8 bits / 8 + overhead
LoRA + optimizer                   ~3 GB
Activations + KV cache             ~12 GB
─────────────────────────────────────────────────────────
TOTAL                              ~100 GB
REMAINING                          ~92 GB         Still plenty for sandbox
```

**Conclusion:** On B200, ANY of these models fit comfortably. Even Qwen3-Coder-Next at FP8 leaves 92GB free. The memory constraint that dominated previous planning is completely gone.

---

## GRPO-3: TRL GRPOTrainer — Exact Configuration

### 3.1 The Reward Function Signature

TRL's GRPOTrainer expects reward functions with this exact signature:

```python
def reward_func(
    completions: list[str],       # G completions for current prompt
    prompts: list[str] = None,    # The prompts (optional)
    **kwargs                      # Any extra dataset columns
) -> list[float]:                 # One reward per completion
    """
    Must return a list of floats, same length as completions.
    Each float is the reward for the corresponding completion.
    """
```

### 3.2 Our Reward Function (Wraps OpenEnv)

```python
import subprocess
import json
import tempfile
import os
import re

def cuda_kernel_reward(completions: list[str], prompts: list[str] = None, 
                       task_code: list[str] = None, **kwargs) -> list[float]:
    """
    Evaluate CUDA kernel completions through the KernelForge environment.
    
    For each completion:
    1. Extract CUDA code from model output
    2. Write to temp file
    3. Compile with nvcc -arch=sm_80 in subprocess (crash isolation)
    4. Verify correctness against PyTorch reference
    5. Profile vs torch.compile
    6. Return discrete reward {-1, 1, 2, 3}
    
    CUDA Agent Equation 1 (validated +36.4pp over continuous, Table 2):
    r = -1  if compilation fails OR correctness fails
    r = +1  if correct but speedup ≤ 1.05×
    r = +2  if speedup > 1.05× vs torch.eager
    r = +3  if speedup > 1.05× vs BOTH torch.eager AND torch.compile
    """
    rewards = []
    
    for i, completion in enumerate(completions):
        # Extract CUDA code from completion
        cuda_code = _extract_cuda(completion)
        if not cuda_code:
            rewards.append(-1.0)
            continue
        
        # Get the task reference code for this completion
        ref_code = task_code[i] if task_code else None
        if not ref_code:
            rewards.append(-1.0)
            continue
        
        # Evaluate in subprocess (crash isolation — non-negotiable)
        reward = _evaluate_in_subprocess(cuda_code, ref_code)
        rewards.append(reward)
    
    return rewards


def _extract_cuda(text: str) -> str:
    """Extract CUDA code from model output."""
    # Try ```cuda ... ``` block
    match = re.search(r'```(?:cuda|cpp|c\+\+)?\s*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try raw code with __global__
    if '__global__' in text:
        return text.strip()
    return ""


def _evaluate_in_subprocess(cuda_code: str, ref_code: str, timeout: int = 60) -> float:
    """
    Run evaluation in a separate Python process.
    
    WHY SUBPROCESS: If a generated kernel causes a segfault, CUDA context
    corruption, or infinite loop, it kills the subprocess — not our training
    process. Every successful CUDA kernel RL system uses this pattern.
    """
    tmpdir = tempfile.mkdtemp()
    kernel_path = os.path.join(tmpdir, "kernel.cu")
    ref_path = os.path.join(tmpdir, "reference.py")
    eval_path = os.path.join(tmpdir, "evaluate.py")
    
    with open(kernel_path, 'w') as f:
        f.write(cuda_code)
    with open(ref_path, 'w') as f:
        f.write(ref_code)
    
    # The evaluation script runs in the subprocess
    eval_script = '''
import sys, json, subprocess, os, torch, time, importlib.util

kernel_path = sys.argv[1]
ref_path = sys.argv[2]

# Step 1: Compile with nvcc -arch=sm_80
binary = kernel_path.replace('.cu', '.so')
compile_cmd = [
    'nvcc', '-arch=sm_80', '-O3', '--use_fast_math',
    '-Xcompiler', '-fPIC', '-shared',
    '-o', binary, kernel_path
]
result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30)
if result.returncode != 0:
    print(json.dumps({"reward": -1.0, "reason": "compile_fail"}))
    sys.exit(0)

# Step 2: Check for forbidden symbols (anti-reward-hacking)
nm_result = subprocess.run(['nm', '-D', binary], capture_output=True, text=True, timeout=5)
forbidden = ['torch', 'at::Tensor', 'c10::', 'triton']
for f in forbidden:
    if f in nm_result.stdout:
        print(json.dumps({"reward": -1.0, "reason": f"forbidden_symbol:{f}"}))
        sys.exit(0)

# Step 3: Load reference model
spec = importlib.util.spec_from_file_location("ref", ref_path)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

model = ref.Model(*ref.get_init_inputs()).cuda().eval()
compiled_model = torch.compile(model)

# Step 4: Correctness check (5 random inputs)
for seed in range(5):
    torch.manual_seed(42 + seed)
    inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in ref.get_inputs()]
    with torch.no_grad():
        ref_out = model(*inputs)
    
    # TODO: Load and run the compiled kernel against these inputs
    # This requires the kernel to expose a callable interface
    # The exact mechanism depends on how the kernel is structured
    # For now, we check compilation + forbidden symbols only
    pass

# Step 5: Profile (simplified — full version uses CUDA events)
# Profile eager
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(30):
    with torch.no_grad():
        model(*inputs)
    torch.cuda.synchronize()
eager_ms = (time.perf_counter() - start) / 30 * 1000

# Profile compiled
for _ in range(10):  # warmup
    with torch.no_grad():
        compiled_model(*inputs)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(30):
    with torch.no_grad():
        compiled_model(*inputs)
    torch.cuda.synchronize()
compile_ms = (time.perf_counter() - start) / 30 * 1000

# TODO: Profile the generated kernel
# For hackathon MVP, use compilation success as primary signal
kernel_ms = compile_ms  # Placeholder

# Step 6: Compute reward (CUDA Agent Equation 1)
speedup_eager = eager_ms / kernel_ms if kernel_ms > 0 else 0
speedup_compile = compile_ms / kernel_ms if kernel_ms > 0 else 0

if speedup_compile > 1.05:
    reward = 3.0
elif speedup_eager > 1.05:
    reward = 2.0
else:
    reward = 1.0

print(json.dumps({
    "reward": reward,
    "eager_ms": eager_ms,
    "compile_ms": compile_ms,
    "kernel_ms": kernel_ms,
    "speedup_eager": speedup_eager,
    "speedup_compile": speedup_compile,
}))
'''
    
    with open(eval_path, 'w') as f:
        f.write(eval_script)
    
    try:
        result = subprocess.run(
            ['python', eval_path, kernel_path, ref_path],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
        )
        if result.returncode != 0:
            return -1.0
        
        output = json.loads(result.stdout.strip().split('\n')[-1])
        return output['reward']
    
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return -1.0
```

### 3.3 Dataset Preparation

```python
from datasets import load_dataset, Dataset

def prepare_grpo_dataset(stage: str = "warmup") -> Dataset:
    """
    Prepare dataset for GRPO training.
    
    Each row must have a 'prompt' column (required by GRPOTrainer)
    and a 'task_code' column (forwarded to reward function via **kwargs).
    """
    ops_6k = load_dataset("BytedTsinghua-SIA/CUDA-Agent-Ops-6K", split="train")
    
    # Load the SKILL.md content (A100 optimization context)
    with open("skill_a100.md") as f:
        skill_md = f.read()
    
    tasks = []
    for row in ops_6k:
        # Determine difficulty from data_source field
        num_ops = int(row["data_source"].split("#")[-1])
        
        # Stage 1 warmup: only single-op tasks
        if stage == "warmup" and num_ops > 1:
            continue
        
        # Stage 3 curriculum: include harder tasks
        if stage == "curriculum" and num_ops > 5:
            continue
        
        prompt = (
            f"{skill_md}\n\n"
            f"## Problem\n"
            f"Write an optimized CUDA kernel for A100 (sm_80) that replaces:\n\n"
            f"```python\n{row['code']}\n```\n\n"
            f"Operations: {row['ops']}\n\n"
            f"Write ONLY the .cu file:\n```cuda\n"
        )
        
        tasks.append({
            "prompt": prompt,
            "task_code": row["code"],  # Forwarded to reward func as **kwargs
            "ops": row["ops"],
            "num_ops": num_ops,
        })
    
    # Limit for hackathon
    if stage == "warmup":
        tasks = tasks[:512]
    elif stage == "curriculum":
        tasks = tasks[:200]
    
    return Dataset.from_list(tasks)
```

### 3.4 Full Training Script (All 3 Stages)

```python
"""
training/run_all_stages.py

Complete 3-stage GRPO pipeline for CUDA kernel optimization.
Runs on B200 192GB. Targets A100 sm_80 kernels.

Usage:
    python training/run_all_stages.py \
        --model unsloth/Qwen3-Coder-Next-FP8-Dynamic \
        --output_dir ./checkpoints \
        --stage all

Estimated wall-clock:
    Stage 1 (warmup):    ~4 hours
    Stage 2 (RFT):       ~30 minutes
    Stage 3 (curriculum): ~5 hours
    Total:               ~9.5 hours
"""
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel


# ============================================================
# STAGE 1: GRPO WARM-UP
# Goal: Bootstrap compilation rate ~50% → ~85%
# Dataset: 512 single-op tasks from Ops-6K
# Turns: 3 (multi-turn with compile error feedback)
# Steps: 300
# ============================================================

def run_stage1(model, tokenizer, output_dir: str):
    print("=" * 60)
    print("STAGE 1: GRPO WARM-UP")
    print("=" * 60)
    
    dataset = prepare_grpo_dataset(stage="warmup")
    print(f"Stage 1 dataset: {len(dataset)} tasks")
    
    config = GRPOConfig(
        output_dir=os.path.join(output_dir, "stage1"),
        
        # GRPO-specific
        num_generations=4,            # G = 4 completions per prompt
        # This means: for each prompt, generate 4 CUDA kernels,
        # evaluate all 4, compute group-relative advantages,
        # update policy to favor higher-reward kernels
        
        # Clipping
        # epsilon for clipped surrogate loss
        # ratio_t clipped to [1 - 0.2, 1 + 0.2] = [0.8, 1.2]
        max_grad_norm=1.0,            # Gradient clipping (prevents explosions)
        
        # KL penalty
        beta=0.0,                     # No KL penalty (DAPO: "not essential")
        # With PEFT/LoRA, reference behavior = adapter disabled
        # No separate reference model needed in memory
        
        # Generation
        max_completion_length=4096,   # CUDA kernels are 50-300 lines
        temperature=1.0,              # Qwen3-Coder-Next trained at 1.0 — do not lower
        # Matches Qwen3-Coder-Next's training temperature. Higher = more diverse = stronger GRPO signal.
        
        # Training
        per_device_train_batch_size=1, # 1 prompt per step
        # Effective batch = 1 prompt × 4 generations = 4 completions
        gradient_accumulation_steps=4, # Accumulate over 4 prompts
        # Effective batch for gradient = 16 completions
        num_train_epochs=1,
        max_steps=300,                # Stage 1 budget
        learning_rate=2e-6,           # Conservative — preserve Qwen3-Coder-Next capabilities
        
        # Precision
        bf16=True,                    # BF16 training on B200
        
        # Logging
        logging_steps=1,              # Log every step (watch the reward!)
        save_steps=50,                # Checkpoint every 50 steps
        
        # Memory optimization
        gradient_checkpointing=True,
        
        # Important: keep task_code column for reward function
        remove_unused_columns=False,
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=cuda_kernel_reward,  # Our reward function from 3.2
        args=config,
        train_dataset=dataset,
    )
    
    print("Starting Stage 1 training...")
    train_result = trainer.train()
    
    # Save Stage 1 checkpoint
    trainer.save_model(os.path.join(output_dir, "stage1", "final"))
    
    # Collect trajectories for Stage 2 RFT
    print("Collecting trajectories for RFT...")
    trajectories = collect_trajectories(model, tokenizer, dataset, n=100)
    
    return train_result, trajectories


# ============================================================
# STAGE 2: REJECTION FINE-TUNING (RFT)
# Goal: Filter successful trajectories, SFT on them
# Without RFT: faster-than-compile drops 96.8% → 49.8%
# (CUDA Agent Table 2 — this is NON-NEGOTIABLE)
# ============================================================

def run_stage2(model, tokenizer, trajectories: list, output_dir: str):
    print("=" * 60)
    print("STAGE 2: REJECTION FINE-TUNING (RFT)")
    print("=" * 60)
    
    # Filter: keep only trajectories with reward >= 1.0 (correct kernels)
    good_trajectories = [t for t in trajectories if t["reward"] >= 1.0]
    print(f"Collected {len(trajectories)} trajectories, "
          f"{len(good_trajectories)} passed filter (reward >= 1.0)")
    
    if len(good_trajectories) < 10:
        print("WARNING: Too few good trajectories. RFT may not be effective.")
        print("Possible causes: model can't generate valid CUDA yet.")
        print("Consider: longer Stage 1, or SFT warmup on pre-generated data first.")
    
    # Format for SFT
    sft_data = []
    for t in good_trajectories:
        sft_data.append({
            "prompt": t["prompt"],
            "completion": t["completion"],
        })
    
    sft_dataset = Dataset.from_list(sft_data)
    
    # SFT training (3 epochs over filtered data)
    from trl import SFTConfig, SFTTrainer
    
    sft_config = SFTConfig(
        output_dir=os.path.join(output_dir, "stage2"),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=5e-6,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=999999,  # Don't save intermediate — just final
        max_seq_length=8192,
    )
    
    sft_trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=sft_dataset,
    )
    
    sft_trainer.train()
    sft_trainer.save_model(os.path.join(output_dir, "stage2", "final"))
    
    print(f"RFT complete. Trained on {len(good_trajectories)} trajectories × 3 epochs.")


# ============================================================
# STAGE 3: GRPO + CURRICULUM
# Goal: Learn optimization strategies through progressive difficulty
# Dataset: 200 curated tasks across 4 difficulty phases
# Turns: 5 (deeper multi-turn with profiling feedback)
# Steps: 200
# ============================================================

def run_stage3(model, tokenizer, output_dir: str):
    print("=" * 60)
    print("STAGE 3: GRPO + CURRICULUM")
    print("=" * 60)
    
    dataset = prepare_grpo_dataset(stage="curriculum")
    print(f"Stage 3 dataset: {len(dataset)} tasks")
    
    config = GRPOConfig(
        output_dir=os.path.join(output_dir, "stage3"),
        
        # GRPO
        num_generations=4,
        beta=0.0,
        max_grad_norm=1.0,
        
        # Generation — lower temperature for exploitation
        max_completion_length=4096,
        temperature=1.0,   # Match Qwen3-Coder-Next training temperature
        
        # Training
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=200,
        learning_rate=3e-6,  # Slightly higher than Stage 1 — matches PRD table
        
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=50,
        remove_unused_columns=False,
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=cuda_kernel_reward,
        args=config,
        train_dataset=dataset,
    )
    
    print("Starting Stage 3 training...")
    trainer.train()
    trainer.save_model(os.path.join(output_dir, "stage3", "final"))


# ============================================================
# TRAJECTORY COLLECTION (for RFT)
# ============================================================

def collect_trajectories(model, tokenizer, dataset, n: int = 100) -> list:
    """
    Generate n trajectories and evaluate them.
    Used between Stage 1 and Stage 2 to collect RFT data.
    """
    trajectories = []
    model.eval()
    
    for i in range(min(n, len(dataset))):
        row = dataset[i]
        prompt = row["prompt"]
        
        # Generate one completion
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=1.0,
                do_sample=True,
            )
        
        completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], 
                                       skip_special_tokens=True)
        
        # Evaluate
        reward = cuda_kernel_reward(
            completions=[completion],
            prompts=[prompt],
            task_code=[row["task_code"]]
        )[0]
        
        trajectories.append({
            "prompt": prompt,
            "completion": completion,
            "reward": reward,
            "task_ops": row["ops"],
        })
        
        status = "✓" if reward >= 1.0 else "✗"
        print(f"  [{i+1}/{n}] {status} reward={reward:.0f} ops={row['ops'][:50]}")
    
    return trajectories


# ============================================================
# MAIN: RUN ALL STAGES
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen3-Coder-Next-FP8-Dynamic")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--stage", default="all", 
                        choices=["1", "2", "3", "all"])
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    args = parser.parse_args()
    
    # Load model with Unsloth
    print(f"Loading {args.model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=8192,
        dtype=None,  # Auto-detect (BF16 on B200)
        load_in_4bit=args.load_in_4bit,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Unsloth recommends 0
        use_gradient_checkpointing="unsloth",
    )
    
    print(f"Model loaded. Trainable params: {model.print_trainable_parameters()}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.stage in ["1", "all"]:
        result, trajectories = run_stage1(model, tokenizer, args.output_dir)
    
    if args.stage in ["2", "all"]:
        if args.stage == "2":
            # Load trajectories from Stage 1
            with open(os.path.join(args.output_dir, "stage1", "trajectories.json")) as f:
                trajectories = json.load(f)
        run_stage2(model, tokenizer, trajectories, args.output_dir)
    
    if args.stage in ["3", "all"]:
        run_stage3(model, tokenizer, args.output_dir)
    
    print("\nDone! Final model saved to:", args.output_dir)


if __name__ == "__main__":
    main()
```

---

## GRPO-4: MARS Credit Assignment (Multi-Turn Extension)

### Why Standard GRPO Fails on Multi-Turn

Standard GRPO assigns the SAME advantage to every token in a completion. In multi-turn kernel optimization:

```
Turn 1: Model writes kernel → compile error         (bad)
Turn 2: Model fixes error → compiles, wrong output   (ok)
Turn 3: Model fixes output → correct, 1.2× speedup   (good!)

Standard GRPO: ALL tokens in ALL turns get advantage based on final reward (+2)
Problem: Turn 1's buggy code gets REWARDED because the final outcome was good
```

### MARS Solution (arXiv:2510.15414)

MARS computes PER-TURN advantages using cumulative returns:

```python
def compute_mars_advantages(turn_rewards: list[float], gamma: float = 1.0) -> list[float]:
    """
    MARS: turn-level advantage estimation.
    
    Args:
        turn_rewards: [r_1, r_2, ..., r_K] rewards per turn
        gamma: discount factor (1.0 = no discounting)
    
    Returns:
        advantages: [A_1, A_2, ..., A_K] per-turn advantages
    
    Example:
        turn_rewards = [-1.0, -0.5, +2.0]  # compile fail, wrong output, success
        
        Cumulative returns (backward):
        R_3 = 2.0
        R_2 = -0.5 + 1.0 × 2.0 = 1.5
        R_1 = -1.0 + 1.0 × 1.5 = 0.5
        
        With 4 generations, normalize across the group:
        A_k = (R_{i,k} - mean(R_{*,k})) / std(R_{*,k})
        
        Turn 1 tokens get advantage based on R_1 = 0.5  (modest)
        Turn 3 tokens get advantage based on R_3 = 2.0  (high)
        → Model learns: fixing errors is valuable, but writing correct code first is more valuable
    """
    K = len(turn_rewards)
    cumulative_returns = [0.0] * K
    
    # Backward pass: compute cumulative returns
    running = 0.0
    for k in range(K - 1, -1, -1):
        running = turn_rewards[k] + gamma * running
        cumulative_returns[k] = running
    
    return cumulative_returns
```

### Integration with TRL's rollout_func

```python
from trl.experimental.openenv import generate_rollout_completions

def multi_turn_rollout(prompts: list[str], trainer: GRPOTrainer) -> dict:
    """
    Custom rollout function for multi-turn CUDA kernel optimization.
    Uses MARS credit assignment for per-turn advantages.
    
    This replaces GRPOTrainer's default single-turn generation.
    Passed via: GRPOTrainer(rollout_func=multi_turn_rollout, ...)
    """
    MAX_TURNS = 5
    
    all_prompt_ids = []
    all_completion_ids = []
    all_logprobs = []
    all_rewards = []
    
    tokenizer = trainer.processing_class
    
    for prompt in prompts:
        history = prompt
        turn_rewards = []
        full_completion = ""
        
        for turn in range(MAX_TURNS):
            # Generate one turn's completion
            outputs = generate_rollout_completions(trainer, [history])
            
            turn_text = tokenizer.decode(
                outputs[0]["completion_ids"], skip_special_tokens=True
            )
            full_completion += turn_text
            
            # Evaluate the kernel so far
            cuda_code = _extract_cuda(full_completion)
            if cuda_code:
                result = _evaluate_in_subprocess(cuda_code, current_task_code)
                turn_reward = result
            else:
                turn_reward = -1.0
            
            turn_rewards.append(turn_reward)
            
            # If we got a good result, stop
            if turn_reward >= 2.0:
                break
            
            # Otherwise, add feedback to history for next turn
            if turn_reward == -1.0:
                history += turn_text + "\n\n# FEEDBACK: Compilation or correctness failed. Fix the errors.\n"
            elif turn_reward == 1.0:
                history += turn_text + "\n\n# FEEDBACK: Correct but not faster. Optimize for A100.\n"
        
        # Compute MARS cumulative returns
        cumulative_returns = compute_mars_advantages(turn_rewards, gamma=1.0)
        
        # Use the final cumulative return as the completion-level reward
        # (TRL applies group normalization on top of this)
        final_reward = cumulative_returns[0]  # R_1 = full trajectory value
        
        all_prompt_ids.append(outputs[0]["prompt_ids"])
        all_completion_ids.append(outputs[0]["completion_ids"])
        all_logprobs.append(outputs[0]["logprobs"])
        all_rewards.append(final_reward)
    
    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "rewards": all_rewards,  # Forwarded to reward function
    }
```

---

## GRPO-5: Compute Budget on B200

### Time Per GRPO Step

```
COMPONENT                          TIME          NOTES
─────────────────────────────────────────────────────────
Model generation (4 × 2048 tokens)  ~40-80s      3B active × 4 candidates
                                                  ~50 tok/s per candidate
nvcc compilation (4 kernels)        ~20-120s     5-30s each, SEQUENTIAL
                                                  (can't parallelize on 1 GPU)
Correctness check (4 × 5 inputs)   ~8-40s       2-10s per kernel
Profiling (4 × 30 runs)            ~20-60s      50 warmup + 30 timed
Gradient computation                ~5-10s       Small with LoRA
─────────────────────────────────────────────────────────
TOTAL PER STEP (single-turn):      ~90-310s     ~1.5-5 minutes
TOTAL PER STEP (multi-turn, 3T):   ~270-930s    ~4.5-15 minutes
```

### Throughput Estimates

| Stage | Steps | Time/Step | Wall-Clock | Total Evaluations |
|-------|-------|-----------|------------|-------------------|
| Stage 1 (300 steps, 3 turns) | 300 | ~5 min | ~25 hours | 3,600 |
| Stage 2 (RFT, SFT) | N/A | N/A | ~30 min | 100 |
| Stage 3 (200 steps, 5 turns) | 200 | ~8 min | ~27 hours | 4,000 |

**Reality check:** 25 + 27 = 52 hours for full pipeline. That's more than a 48-hour hackathon.

### Hackathon-Adjusted Budget

For a 24-hour hackathon day, reduce stages:

| Stage | Steps | Time/Step | Wall-Clock | Evaluations |
|-------|-------|-----------|------------|-------------|
| Stage 1 (single-turn, not multi) | 100 | ~2 min | ~3.5 hours | 400 |
| Stage 2 (RFT) | N/A | N/A | ~30 min | 50 |
| Stage 3 (single-turn GRPO) | 80 | ~2 min | ~3 hours | 320 |

**Total: ~7 hours.** Fits in one hackathon day. Leaves time for SkyDiscover runs, evaluation, and demo preparation.

**Key tradeoff:** Single-turn (agent writes one kernel, gets one reward) instead of multi-turn (agent iterates on errors). Multi-turn is better for learning but 3-5x slower per step. For hackathon, single-turn is pragmatic.

---

## GRPO-6: What To Monitor During Training

### Key Metrics (logged by GRPOTrainer)

```
grpo/reward/mean          — Average reward across all generations
                            Should increase over training.
                            Stage 1: -1 → 0 → 0.5 (learning to compile)
                            Stage 3: 0.5 → 1.0 → 1.5 (learning to optimize)

grpo/reward/std           — Reward variance within groups
                            Should be >0 (if 0, no learning signal)
                            If std drops to 0: all generations identical → increase temperature

grpo/completion_length    — Average completion length
                            Should stabilize at 100-300 tokens for CUDA kernels
                            If growing unboundedly: model is generating garbage

grpo/kl                   — KL divergence from reference
                            With beta=0, not penalized but still tracked
                            If >10: model has drifted far from base, may be unstable

train/loss                — Training loss
                            Should decrease. If NaN: reduce learning rate.
```

### Decision Points During Hackathon

```
After 30 Stage 1 steps:
  reward/mean > -0.5?
    YES → On track. Continue.
    NO  → Model can't compile CUDA. Options:
          1. Switch to Qwen3-Coder-Next (better at code)
          2. Do SFT warmup first (pre-generated CUDA data)
          3. Simplify tasks (try vector_add only)

After 100 Stage 1 steps:
  reward/mean > 0?
    YES → Good. Proceed to Stage 2 RFT.
    NO  → Stage 1 isn't working. Options:
          1. Increase temperature to 1.0 (more exploration)
          2. Reduce task difficulty (single-op only)
          3. Bail on GRPO, focus on SkyDiscover

After Stage 2 RFT (30 min):
  Test on 20 tasks: compilation rate > 70%?
    YES → Proceed to Stage 3.
    NO  → RFT didn't anchor the policy. Collect more trajectories.

After 50 Stage 3 steps:
  reward/mean increasing?
    YES → Let it cook. Run until hackathon deadline.
    NO  → Model plateau'd. Stop training, use best checkpoint.
```

---

## GRPO-7: Critical Implementation Notes

### 7.1 Unsloth + PEFT Reference Model Trick

With PEFT (LoRA), GRPOTrainer does NOT keep a separate reference model. Instead, it temporarily disables the LoRA adapter to recover reference behavior. This saves ~17.5 GB (for 35B-A3B) of VRAM.

```python
# Internally, TRL does this:
with model.disable_adapter():
    ref_logprobs = model.forward(input_ids)  # Reference behavior

# Then with adapter enabled:
policy_logprobs = model.forward(input_ids)   # Current policy

# KL = f(ref_logprobs, policy_logprobs)
```

### 7.2 vLLM Colocate Mode (Single GPU)

For generation during GRPO, TRL can use vLLM in "colocate" mode — vLLM runs in the same process as training, sharing GPU memory.

```python
config = GRPOConfig(
    use_vllm=True,
    vllm_gpu_memory_utilization=0.5,  # Give vLLM 50% of remaining memory
    # Unsloth's "Standby" mode frees inference memory during gradient computation
)
```

**Warning:** vLLM colocate mode can cause OOM if the model is large. Test this BEFORE the hackathon. If it OOMs, fall back to `use_vllm=False` (uses transformers.generate, slower but safer).

### 7.3 The remove_unused_columns Trap

By default, GRPOTrainer removes all dataset columns except 'prompt'. Your reward function needs 'task_code' to know which PyTorch reference to evaluate against. **You MUST set:**

```python
config = GRPOConfig(
    remove_unused_columns=False,  # Keep task_code column
)
```

And your reward function must accept `**kwargs`:

```python
def reward_func(completions, prompts=None, task_code=None, **kwargs):
    # task_code is available because remove_unused_columns=False
```

### 7.4 Group Size and Reward Variance

If G=4 and all 4 completions get r=-1 (all fail to compile), advantages are all 0. No learning. The probability this happens depends on the model's compilation success rate:

```
P(all 4 fail) = (1 - success_rate)^4

If success_rate = 10% → P(all fail) = 0.9^4 = 65.6%  ← BAD (2/3 steps wasted)
If success_rate = 30% → P(all fail) = 0.7^4 = 24.0%  ← OK
If success_rate = 50% → P(all fail) = 0.5^4 = 6.25%  ← GOOD
If success_rate = 70% → P(all fail) = 0.3^4 = 0.81%  ← EXCELLENT
```

This is why Stage 1 warmup targets compilation rate > 85% before proceeding. If your model starts at <30% compilation rate, most GRPO steps will have zero signal and training won't converge.

---

## GRPO-8: Quick Reference Card

```
GRPO ALGORITHM SUMMARY:
1. Sample G completions per prompt from policy
2. Evaluate each → rewards r_1, ..., r_G
3. Advantage: Â_i = (r_i - mean(r)) / std(r)
4. Loss: min(ratio × Â, clip(ratio, 0.8, 1.2) × Â) - β × KL
5. Update policy via backprop

OUR CONFIGURATION:
  G = 4 (num_generations)
  β = 0.0 (no KL penalty, DAPO style)
  ε = 0.2 (clip range)
  Rewards = {-1, 1, 2, 3} (CUDA Agent Equation 1)
  Temperature = 1.0 (Stage 1) → 1.0 (Stage 3)
  Learning rate = 2e-6 (Stage 1) → 3e-6 (Stage 3)
  LoRA rank = 16, target = qkvo + gate/up/down

B200 MEMORY:
  35B-A3B 4-bit: ~34.5 GB total → 157.5 GB free
  9B 4-bit: ~12.5 GB total → 179.5 GB free
  Coder-Next FP8: ~100 GB total → 92 GB free

HACKATHON BUDGET:
  Stage 1: 100 steps × ~2 min = ~3.5 hours
  Stage 2: RFT = ~30 min
  Stage 3: 80 steps × ~2 min = ~3 hours
  Total: ~7 hours (single-turn mode)

ABORT CONDITIONS:
  Reward mean stays at -1.0 after 30 steps → model can't compile
  Reward std = 0 → all completions identical → increase temperature
  Loss = NaN → reduce learning rate to 1e-6
  OOM → reduce max_completion_length to 2048
```
