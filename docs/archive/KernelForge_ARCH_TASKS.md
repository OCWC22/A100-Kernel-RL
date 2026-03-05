# KernelForge Architectural Task Document

**Generated:** March 4, 2026
**Purpose:** Complete technical synthesis for hackathon RL readiness
**Sources:** PRD v3/v4/v5 + Implementation Spec + Truth Document + Codebase Audit

---

## Executive Summary

KernelForge is an RL post-training system that teaches Qwen3-Coder-Next (80B MoE, ~3.9B active) to generate optimized CUDA kernels targeting A100 (sm_80). It combines:

- **ByteDance CUDA Agent** methodology (arXiv:2602.24286): 3-stage pipeline proven to prevent policy collapse
- **DoubleAI WarpSpeed/DoubleGraph** patterns: Per-GPU specialization, CachePool, 4-way dispatch curriculum

**Current State:** Core infrastructure exists but has critical bugs, architectural mismatches, and 13 missing files. This document provides the complete task breakdown to achieve hackathon readiness.

---

## Part 1: Architecture Overview

### 1.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KERNELFORGE SYSTEM ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   STAGE 1   │───▶│   STAGE 2   │───▶│   STAGE 3   │───▶│  EVALUATION │    │
│  │ GRPO Warmup │    │     RFT     │    │GRPO+Curric. │    │   & Demo    │    │
│  │  (60 steps) │    │ (100 steps) │    │  (60 steps) │    │             │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    │
│         │                  │                  │                  │           │
│         ▼                  ▼                  ▼                  ▼           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        OPENENV ENVIRONMENT                           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │  Reset   │─▶│  Step    │─▶│  Reward  │─▶│  State   │            │   │
│  │  │(SKILL.md)│  │(CUDA src)│  │{-1,1,2,3}│  │(metrics) │            │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │   │
│  │       │             │             │             │                   │   │
│  │       ▼             ▼             ▼             ▼                   │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │              LOCAL COMPILE-VERIFY-BENCHMARK LOOP            │    │   │
│  │  │  nvcc -arch=sm_80 → correctness test → timing benchmark     │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          SUPPORTING INFRA                            │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │   │
│  │  │ CachePool  │  │ Anti-Hack  │  │  PAC Verify │  │   Modal    │    │   │
│  │  │  (LRU 8)   │  │(symbol scan│  │ (3 invars) │  │  (backup)  │    │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Target Hardware

| Component | Specification | Source |
|-----------|--------------|--------|
| **Target GPU (kernels)** | A100 (sm_80) | Truth Part 0 (LOCKED) |
| **Training GPU** | H100 80GB (primary) or B200 192GB (fallback) | Truth Part 0 |
| **A100 L2 Cache** | 40 MB | NVIDIA whitepaper |
| **A100 SMEM per SM** | 164 KB | NVIDIA whitepaper |
| **A100 SMs** | 108 | NVIDIA whitepaper |
| **A100 HBM Bandwidth** | 2,039 GB/s (SXM) | NVIDIA whitepaper |
| **nvcc arch flag** | `-arch=sm_80` | Truth Part 15 |

### 1.3 Model Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| **Primary Model** | Qwen3-Coder-Next (80B MoE, ~3.9B active) | Truth Part 0 |
| **Fallback Model** | Qwen2.5-Coder-7B-Instruct | Truth Part 0 |
| **GPTQ Model ID** | dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16 | Truth Part 15 |
| **GPTQ VRAM (weights)** | ~43 GB | Truth Part 15 |
| **Training VRAM (H100)** | ~54 GB total | Truth Part 15 |
| **LoRA Rank** | 16 | Truth Part 15 |
| **LoRA Alpha** | 16 | Truth Part 15 |
| **LoRA Targets (Qwen3)** | q/k/v/o_proj + shared_expert gate/up/down | Truth Part 15 |
| **LoRA Targets (Qwen2.5)** | q/k/v/o/gate/up/down_proj | Truth Part 15 |

### 1.4 Training Pipeline Constants

| Parameter | Stage 1 | Stage 2 | Stage 3 | Source |
|-----------|---------|---------|---------|--------|
| **Algorithm** | GRPO | SFT | GRPO | Truth Part 8 |
| **max_steps** | 100 | 100 | 60 | Truth Part 15 |
| **learning_rate** | 3e-6 | 5e-6 | 3e-6 | Truth Part 8 |
| **temperature** | 0.9 | N/A | 0.7 | Truth Part 15 |
| **num_generations** | 4-8 | N/A | 4-8 | Truth Part 15 |
| **beta (KL)** | 0.0 | N/A | 0.0 | Truth Part 15 |
| **max_completion_length** | 4096 | 4096 | 4096 | Truth Part 15 |

### 1.5 Reward Function (CUDA Agent Equation 1)

```
r = -1   if compilation fails OR correctness fails
r = +1   if correct but no speedup (speedup ≤ 1.05×)
r = +2   if >5% faster than torch.eager
r = +3   if >5% faster than BOTH torch.eager AND torch.compile
```

**Evidence:** Discrete rewards beat continuous speedup by 36.4 percentage points (CUDA Agent Table 2).

---

## Part 2: Document Evolution Summary

### 2.1 Decision Evolution Across PRD Versions

| Dimension | PRD v3 | PRD v4 | PRD v5 | Truth |
|-----------|--------|--------|--------|-------|
| **Target GPU** | H100 SXM5 (sm_90a) | A100 (sm_80) | A100 kernels; H100/B200 training | A100 (sm_80) only |
| **Model** | Qwen3-Coder-Next 80B MoE | Qwen2.5-Coder-7B | Qwen3-Coder-Next 80B MoE | Qwen3-Coder-Next + Qwen2.5 fallback |
| **Training** | TRL GRPOTrainer + Unsloth | GRPO (no critic) | Multi-stage: RL→RFT→GRPO | 3-stage: Warm-up→RFT→GRPO |
| **Environment** | Custom OpenEnv on Modal H100 | OpenEnv (openenv-core) | Multi-algorithm with curriculum | OpenEnv ≥0.2.1 |
| **Kernel Scope** | WCC (Union-Find) | WCC (focused) | Multi-algorithm: WCC+fusion+ops | Multi-algorithm: Ops-6K subset |
| **Reward** | Simple 4-tier | CUDA Agent validated | CUDA Agent + DoubleGraph | {-1,1,2,3} milestones |
| **Anti-Hack** | Basic | From CUDA Agent §3.2 | Full suite | Full suite |

### 2.2 Key Corrections Across Versions

| Prior Error | Final Correction (Truth) |
|-------------|--------------------------|
| "NO SFT JUST RL" — pure RL from base model | Multi-stage: warm-up → RFT → GRPO (CUDA Agent proved pure RL collapses at step 17) |
| DoubleGraph as reference only | DoubleGraph patterns productized into curriculum, SKILL.md, CachePool, .cu.flags |
| Qwen2.5-Coder-7B as primary | Qwen3-Coder-Next primary, Qwen2.5 as fallback |
| WCC-only scope | Multi-algorithm from Ops-6K (200-problem curated subset) |
| Simple reward | CUDA Agent's validated {-1, 1, 2, 3} with ablation evidence |
| No anti-hacking | Full suite from CUDA Agent Section 3.2 |
| No curriculum | DoubleGraph-inspired 4-phase progression |
| No scarce data analysis | Dedicated section with 5-layer mitigation stack |

---

## Part 3: Current Codebase State

### 3.1 What EXISTS and Is Aligned

| Component | File Path | Status | Notes |
|-----------|-----------|--------|-------|
| Anti-hack utilities | `openenv_env/anti_hack.py` | ✅ Aligned | CU_FLAGS whitelist, forbidden symbol scanning |
| GPU CachePool | `openenv_env/cache_pool.py` | ✅ Aligned | LRU, 8 entries, cleanup on eviction |
| Reward function | `openenv_env/reward.py` | ✅ Aligned | Canonical 4-level {-1,1,2,3} milestones |
| Local OpenEnv environment | `openenv_env/kernelforge_env.py` | ✅ Aligned | Local compile→verify→benchmark, no Modal in step |
| GPU registry | `openenv_env/kernelforge_env.py` | ✅ Aligned | a100/h100/b200 with correct specs |
| CUDA-Agent integration | `training/cuda_agent_integration.py` | ✅ Aligned | Ops-6K loading, prompt building |
| PAC verification | `verification/pac_verify.py` | ✅ Aligned | 3 invariants, 5 adversarial graph types |
| Modal evaluation backend | `modal_app.py` | ✅ Aligned | evaluate_kernel with full pipeline |
| Baseline profiling | `modal_app.py` profile_baselines | ✅ Aligned | Compiled kernel → CPU fallback chain |
| A100 defaults | Environment variables | ✅ Aligned | KERNELFORGE_TARGET_GPU=A100, sm_80 |
| Status documentation | `A100_HACKATHON_STATUS.md` | ✅ Aligned | Honest gap tracking |
| SPAN alignment doc | `docs/KernelForge_Updated_SPAN.md` | ✅ Aligned | Decision evolution and gaps |

### 3.2 What EXISTS but DIFFERS from Truth

| Issue | File | Line(s) | Truth Spec | Current Code | Impact |
|-------|------|---------|------------|--------------|--------|
| **P0-1: Module-level model loading** | `training/stage3_grpo.py` | 33-53 | Lazy loading inside main() | Executes on import | Unimportable without 80B model |
| **P0-2: LoRA targets all 512 experts** | `training/stage3_grpo.py` | 45-51 | Attention + shared_expert only | All experts included | Won't fit in VRAM (~207GB) |
| **P0-2b: LoRA rank wrong** | `training/stage3_grpo.py` | 45 | r=16, alpha=16 | r=32 | Doubles adapter memory |
| **P0-3: RFT filter threshold wrong** | `training/rft_filter.py` | 285 | min_reward=1.0 | min_reward=2.0 | Discards correct trajectories |
| **P0-4: ZeroDivisionError** | `training/rft_filter.py` | 285-288 | Guard for empty trajectories | No guard | Crashes on first run |
| **P0-5: Wrong nvcc arch** | `verification/profile.py` | 346 | -arch=sm_80 | -arch=sm_90a | Fails on A100 |
| **P1-7: Stage 1 is SFT not GRPO** | `training/stage1_warmup.py` | entire file | GRPO warm-up | SFTTrainer | No RL signal in Stage 1 |
| **P1-8: Reward duplicated 4x** | Multiple files | Various | Single compute_reward() | 4 implementations | Semantic drift risk |
| **P1-11: WCC-only training data** | `training/stage3_grpo.py` | load_wcc_dataset | Ops-6K multi-algorithm | WCC Union-Find only | Wrong kernel scope |
| **P1-13: SKILL.md minimal** | `skill_a100.md` | all | Full optimization library | WCC-only | Missing optimization techniques |
| **P1-15: RFT missing SFT step** | `training/rft_filter.py` | main() | Filter then SFT | Only saves dataset | Incomplete RFT stage |

### 3.3 What is MISSING Entirely (Truth Part 13)

| Expected Path | Purpose | Impact if Missing |
|---------------|---------|-------------------|
| `training/model_loader.py` | Qwen3/Qwen2.5 fallback selection | No automated recovery on model load failure |
| `training/stage1_warmup.py` | GRPO warm-up entrypoint | Stage 1 uses wrong algorithm (SFT) |
| `training/stage2_rft.py` | Complete RFT with SFT | RFT incomplete |
| `training/stage3_grpo.py` | GRPO with curriculum | Naming mismatch, no curriculum |
| `training/curriculum.py` | Progressive difficulty management | No curriculum logic |
| `datasets/download_ops6k.py` | Pull CUDA-Agent-Ops-6K | Manual dataset setup |
| `datasets/curate_subset.py` | 200-problem subset curation | No curated dataset |
| `datasets/compute_baselines.py` | Pre-compute eager/compile times | No baseline timings |
| `evaluation/eval_model.py` | Full evaluation suite | Cannot measure model quality |
| `evaluation/ablation.py` | H1/H2/H3 hypothesis tests | Cannot validate claims |
| `tests/test_env.py` | Environment unit tests | No env validation |
| `tests/test_reward.py` | Reward function tests | No reward validation |
| `tests/test_compile.py` | sm_80 compilation tests | No compilation validation |

---

## Part 4: Detailed Task Breakdown

### 4.1 P0 — Critical Blockers (Must Fix Before Any Run)

#### Task P0-1: Fix Module-Level Model Loading
**File:** `training/stage3_grpo.py`
**Lines:** 33-53
**Problem:** Model loading executes on import, making module unimportable without 80B model
**Fix:**
```python
# Move all model loading into main() or a lazy-init function
def get_model_and_tokenizer():
    """Lazy model loading - only called when needed."""
    model, tokenizer = FastModel.from_pretrained(...)
    model = FastLanguageModel.get_peft_model(model, ...)
    return model, tokenizer

def main():
    model, tokenizer = get_model_and_tokenizer()
    # ... rest of training
```

#### Task P0-2: Fix LoRA Targets for MoE
**File:** `training/stage3_grpo.py`, `training/stage1_warmup.py`
**Lines:** grpo_train.py:45-51, sft_warmup.py:37-43, 281-287
**Problem:** LoRA targets all 512 routed experts (~207GB optimizer states)
**Fix:**
```python
# Correct targets for Qwen3-Coder-Next MoE
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",       # Attention layers
    "shared_expert.gate_proj",                      # Shared expert only
    "shared_expert.up_proj",
    "shared_expert.down_proj",
],
lora_rank=16,        # Was 32
lora_alpha=16,       # Keep alpha = rank
```

#### Task P0-3: Fix RFT Filter Threshold
**File:** `training/rft_filter.py`
**Line:** 285
**Problem:** min_reward=2.0 discards correct-but-not-fast trajectories
**Fix:**
```python
def filter_trajectories(self, min_reward: float = 1.0) -> List[Dict[str, Any]]:
    # Keep all correct kernels (reward >= 1.0), not just fast ones
```

#### Task P0-4: Add Empty Trajectory Guard
**File:** `training/rft_filter.py`
**Lines:** 285-288
**Problem:** ZeroDivisionError when no trajectories collected
**Fix:**
```python
total = len(self.trajectories)
pct = (len(filtered) / total * 100) if total > 0 else 0.0
print(f"Filtered {len(filtered)}/{total} trajectories ({pct:.1f}%)")
```

#### Task P0-5: Fix nvcc Arch in Profiler
**File:** `verification/profile.py`
**Line:** 346
**Problem:** Compiles for sm_90a (H100) but target is sm_80 (A100)
**Fix:**
```python
arch = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
compile_cmd = ["nvcc", f"-arch={arch}", "-O3", ...]
```

---

### 4.2 P1 — Architectural Mismatches (Should Fix)

#### Task P1-7: Convert Stage 1 from SFT to GRPO Warm-up
**File:** `training/stage1_warmup.py` → rename to `training/stage1_warmup.py`
**Problem:** Stage 1 uses SFTTrainer but Truth specifies GRPO warm-up
**Fix:** Rewrite as GRPO training:
```python
# stage1_warmup.py - GRPO warm-up (NOT SFT)
from trl import GRPOTrainer, GRPOConfig

def main():
    model, tokenizer = load_model()  # Lazy load
    dataset = load_easy_operators()   # 50 single-op problems
    
    training_args = GRPOConfig(
        output_dir="./outputs/stage1",
        learning_rate=3e-6,
        temperature=0.9,          # High for exploration
        max_steps=100,
        num_generations=4,
        beta=0.0,                 # No KL penalty
        max_completion_length=4096,
    )
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[cuda_kernel_reward],
        train_dataset=dataset,
    )
    trainer.train()
```

#### Task P1-8: Consolidate Reward Computation
**Files:** `training/stage3_grpo.py`, `training/rft_filter.py`, `openenv_env/kernel_forge_env.py`
**Problem:** Reward logic duplicated 4 times with different baseline terminology
**Fix:** All locations should import and call `compute_reward()` from `openenv_env/reward.py`:
```python
from openenv_env.reward import compute_reward

# In cuda_kernel_reward():
result = eval_fn.remote(...)
reward = compute_reward(
    compiles=result["compiles"],
    correct=result["correct"],
    speedup_vs_eager=result.get("speedup_vs_orig", 0),
    speedup_vs_compile=result.get("speedup_vs_dg", 0),
)
```

#### Task P1-9: Create Curriculum Module
**File:** `training/curriculum.py` (new)
**Problem:** No progressive difficulty management
**Implementation:**
```python
"""
Curriculum management for KernelForge training.

4-Phase progression (from DoubleGraph 4-way dispatch):
  Phase A: Single operators (easy)
  Phase B: 2-op fusion (medium)
  Phase C: Architecture-specific optimizations (hard)
  Phase D: Advanced patterns (expert)

Promotion rule: >50% of last 10 steps achieve target reward
Demotion rule: <20% of last 10 steps achieve any positive reward
"""
from dataclasses import dataclass
from typing import List, Dict
import random

@dataclass
class CurriculumPhase:
    name: str
    difficulty: str
    problems: List[Dict]
    target_reward: float
    min_success_rate: float = 0.5

class CurriculumManager:
    PHASES = ["A", "B", "C", "D"]
    
    def __init__(self, dataset: List[Dict]):
        self.phases = self._partition_by_difficulty(dataset)
        self.current_phase = "A"
        self.reward_history: List[float] = []
    
    def get_next_prompt(self) -> Dict:
        """Sample from current phase, weighted by success rate."""
        phase = self.phases[self.current_phase]
        # PRIME technique: maintain 20-80% success rate per problem
        eligible = [p for p in phase.problems 
                    if 0.2 <= p.get("success_rate", 0.5) <= 0.8]
        return random.choice(eligible) if eligible else random.choice(phase.problems)
    
    def update_reward(self, reward: float):
        """Record reward and check for phase transition."""
        self.reward_history.append(reward)
        if len(self.reward_history) >= 10:
            recent = self.reward_history[-10:]
            success_rate = sum(1 for r in recent if r >= self.phases[self.current_phase].target_reward) / 10
            
            if success_rate >= 0.5 and self.current_phase != "D":
                self._promote()
            elif success_rate < 0.2 and self.current_phase != "A":
                self._demote()
    
    def _promote(self):
        idx = self.PHASES.index(self.current_phase)
        if idx < len(self.PHASES) - 1:
            self.current_phase = self.PHASES[idx + 1]
            self.reward_history.clear()
    
    def _demote(self):
        idx = self.PHASES.index(self.current_phase)
        if idx > 0:
            self.current_phase = self.PHASES[idx - 1]
            self.reward_history.clear()
```

#### Task P1-10: Create Model Loader with Fallback
**File:** `training/model_loader.py` (new)
**Problem:** No fallback path from Qwen3-Coder-Next to Qwen2.5-Coder-7B
**Implementation:**
```python
"""
Model loader with automatic fallback.

Decision tree (Truth Part 7.6):
1. Try Qwen3-Coder-Next GPTQ-Int4 (~43GB weights)
2. If load fails → fall back to Qwen2.5-Coder-7B-Instruct
3. Attach LoRA with architecture-appropriate targets
"""
import os
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# LoRA targets per model (Truth Part 15)
QWEN3_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj",
]
QWEN25_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def load_model_with_fallback(
    primary_model: str = "dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16",
    fallback_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    lora_rank: int = 16,
    lora_alpha: int = 16,
) -> Tuple[object, object, str]:
    """
    Load model with automatic fallback.
    
    Returns: (model, tokenizer, model_type)
    """
    # Try primary (Qwen3-Coder-Next GPTQ)
    try:
        print(f"Attempting to load {primary_model}...")
        model = AutoModelForCausalLM.from_pretrained(
            primary_model,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(primary_model)
        
        # Attach LoRA for MoE
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=QWEN3_TARGETS,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        return model, tokenizer, "qwen3_moe"
    
    except Exception as e:
        print(f"Primary model failed: {e}")
        print(f"Falling back to {fallback_model}...")
    
    # Fallback (Qwen2.5-Coder-7B)
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=fallback_model,
        max_seq_length=8192,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=QWEN25_TARGETS,
        lora_dropout=0,
        bias="none",
    )
    return model, tokenizer, "qwen25_dense"
```

#### Task P1-11: Decouple from WCC-Only Training Data
**Files:** `training/stage3_grpo.py`, `training/stage1_warmup.py`, `training/rft_filter.py`
**Problem:** Fallback datasets are WCC Union-Find only, Truth specifies multi-algorithm Ops-6K
**Fix:**
1. Rename `load_wcc_dataset()` → `load_ops6k_dataset()`
2. Update `create_minimal_dataset()` to use diverse operators from Ops-6K
3. Update `generate_wcc_examples()` → `generate_operator_examples()` with multiple algorithm types

#### Task P1-13: Expand SKILL.md
**File:** `skill_a100.md`
**Problem:** Only covers WCC Union-Find, missing most optimization techniques
**Fix:** Replace with full SKILL.md from Truth Part 6 (100+ lines covering algebraic reduction, memory hierarchy, compute optimization, library integration)

#### Task P1-15: Add SFT Step to RFT
**File:** `training/rft_filter.py` → `training/stage2_rft.py`
**Problem:** RFT only saves filtered dataset, doesn't run SFT
**Fix:**
```python
# In main(), after filtering:
if len(filtered_trajectories) > 0:
    # Convert to SFT dataset format
    sft_dataset = format_for_sft(filtered_trajectories)
    
    # Run SFT training
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir="./outputs/stage2_rft",
            max_steps=100,
            learning_rate=5e-6,
        ),
        train_dataset=sft_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model("./outputs/stage2_checkpoint")
```

---

### 4.3 P2 — Missing Files (Create New)

#### Task P2-1: Create Dataset Utilities

**File:** `datasets/download_ops6k.py`
```python
"""Download CUDA-Agent-Ops-6K from HuggingFace."""
from datasets import load_dataset
import json
from pathlib import Path

def download_ops6k(output_dir: str = "./data/ops6k"):
    """Download and cache Ops-6K dataset."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    ds = load_dataset("BytedTsinghua-SIA/CUDA-Agent-Ops-6K", split="train")
    
    # Save as JSONL for easy loading
    with open(f"{output_dir}/ops6k.jsonl", "w") as f:
        for item in ds:
            f.write(json.dumps(item) + "\n")
    
    print(f"Downloaded {len(ds)} operators to {output_dir}")
    return ds

if __name__ == "__main__":
    download_ops6k()
```

**File:** `datasets/curate_subset.py`
```python
"""Curate 200-problem subset from Ops-6K."""
import json
import random
from pathlib import Path

def curate_subset(
    input_path: str = "./data/ops6k/ops6k.jsonl",
    output_path: str = "./data/curated_200.jsonl",
    n_easy: int = 50,
    n_medium: int = 75,
    n_hard: int = 75,
):
    """
    Select balanced subset:
    - Easy: single operators
    - Medium: 2-op fusion (83.77% of Ops-6K)
    - Hard: 3+ op fusion
    """
    with open(input_path) as f:
        all_ops = [json.loads(line) for line in f]
    
    # Classify by complexity
    single_op = [op for op in all_ops if len(op.get("operators", [])) == 1]
    two_op = [op for op in all_ops if len(op.get("operators", [])) == 2]
    multi_op = [op for op in all_ops if len(op.get("operators", [])) >= 3]
    
    # Sample
    easy = random.sample(single_op, min(n_easy, len(single_op)))
    medium = random.sample(two_op, min(n_medium, len(two_op)))
    hard = random.sample(multi_op, min(n_hard, len(multi_op)))
    
    curated = easy + medium + hard
    random.shuffle(curated)
    
    with open(output_path, "w") as f:
        for op in curated:
            f.write(json.dumps(op) + "\n")
    
    print(f"Curated {len(curated)} problems: {len(easy)} easy, {len(medium)} medium, {len(hard)} hard")
    return curated

if __name__ == "__main__":
    curate_subset()
```

**File:** `datasets/compute_baselines.py`
```python
"""Pre-compute torch.eager and torch.compile baseline timings."""
import json
import time
import torch
from pathlib import Path

def compute_baselines(
    dataset_path: str = "./data/curated_200.jsonl",
    output_path: str = "./data/baselines.jsonl",
    n_runs: int = 30,
    warmup: int = 10,
):
    """
    For each operator:
    1. Instantiate PyTorch module
    2. Generate test inputs
    3. Time torch.eager execution
    4. Time torch.compile execution
    """
    with open(dataset_path) as f:
        ops = [json.loads(line) for line in f]
    
    baselines = []
    for op in ops:
        try:
            # Import and instantiate operator module
            # (This requires the operator to be importable)
            # ... implementation depends on Ops-6K format
            
            result = {
                "op_id": op.get("id"),
                "eager_time_ms": None,  # Will be filled
                "compile_time_ms": None,
            }
            
            # Time eager
            # Time compiled
            # ... actual timing code
            
            baselines.append(result)
        except Exception as e:
            print(f"Failed to benchmark {op.get('id')}: {e}")
    
    with open(output_path, "w") as f:
        for b in baselines:
            f.write(json.dumps(b) + "\n")
    
    return baselines

if __name__ == "__main__":
    compute_baselines()
```

#### Task P2-2: Create Evaluation Scripts

**File:** `evaluation/eval_model.py`
```python
"""Full evaluation suite for trained models."""
import json
from pathlib import Path
from typing import Dict, List
import torch

def evaluate_model(
    model,
    tokenizer,
    dataset_path: str = "./data/curated_200.jsonl",
    n_samples: int = 50,
    output_dir: str = "./outputs/evaluation",
):
    """
    Evaluate model on held-out operators.
    
    Metrics:
    - Compilation rate
    - Correctness rate
    - Speedup distribution (vs eager, vs compile)
    - Average reward
    """
    # Load evaluation subset
    # Generate kernels
    # Evaluate each
    # Compute statistics
    
    results = {
        "compilation_rate": 0.0,
        "correctness_rate": 0.0,
        "avg_reward": 0.0,
        "speedup_vs_eager": [],
        "speedup_vs_compile": [],
    }
    
    # ... implementation
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results
```

**File:** `evaluation/ablation.py`
```python
"""
Ablation studies for H1/H2/H3 hypotheses.

H1: Multi-stage RL improves kernel quality over base model.
H2: RFT warm-up is necessary (replicates CUDA Agent's ablation).
H3: DoubleGraph-informed SKILL.md improves over generic prompts.
"""
from dataclasses import dataclass
from typing import Dict, List
import json

@dataclass
class AblationResult:
    hypothesis: str
    condition: str
    metric: str
    value: float
    baseline: float
    delta: float

def run_ablation_h1(model, tokenizer, dataset, env):
    """
    H1: Multi-stage RL improves over base.
    
    Compare:
    - Base model reward distribution
    - Post-Stage-1 reward distribution
    - Post-Stage-2 reward distribution
    - Post-Stage-3 reward distribution
    """
    # ... implementation
    pass

def run_ablation_h2(model, tokenizer, dataset, env):
    """
    H2: RFT is necessary.
    
    Compare:
    - GRPO without RFT (skip Stage 2)
    - GRPO with RFT (full pipeline)
    
    Expected: Without RFT, training collapses (CUDA Agent Table 2).
    """
    # ... implementation
    pass

def run_ablation_h3(model, tokenizer, dataset, env):
    """
    H3: SKILL.md improves over generic prompts.
    
    Compare:
    - Prompts with A100-specific SKILL.md
    - Prompts with generic "write fast CUDA"
    
    Expected: A100 SKILL.md achieves higher speedups.
    """
    # ... implementation
    pass

def run_all_ablations(output_dir: str = "./outputs/ablations"):
    """Run all three ablation studies."""
    results = []
    # ... implementation
    
    with open(f"{output_dir}/ablation_results.json", "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)
    
    return results
```

#### Task P2-3: Create Test Suite

**File:** `tests/test_env.py`
```python
"""Unit tests for OpenEnv environment."""
import pytest
from openenv_env.kernelforge_env import KernelForgeEnv

def test_reset_returns_observation():
    env = KernelForgeEnv()
    obs = env.reset()
    assert "skill_md" in obs
    assert "problem" in obs

def test_step_returns_step_result():
    env = KernelForgeEnv()
    env.reset()
    
    # Simple valid CUDA kernel
    cuda_code = '''
    extern "C" __global__ void simple_add(float* a, float* b, float* c, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) c[i] = a[i] + b[i];
    }
    '''
    
    result = env.step(cuda_code)
    assert hasattr(result, "reward")
    assert hasattr(result, "done")
    assert hasattr(result, "info")

def test_compilation_failure_returns_negative_reward():
    env = KernelForgeEnv()
    env.reset()
    
    # Invalid CUDA code
    bad_code = "this is not valid cuda"
    
    result = env.step(bad_code)
    assert result.reward == -1.0

def test_state_is_serializable():
    env = KernelForgeEnv()
    env.reset()
    state = env.state
    assert isinstance(state, dict)
    # Should be JSON-serializable
    import json
    json.dumps(state)  # Should not raise
```

**File:** `tests/test_reward.py`
```python
"""Unit tests for reward function."""
import pytest
from openenv_env.reward import compute_reward

def test_compile_failure_returns_negative_one():
    reward = compute_reward(compiles=False, correct=False)
    assert reward == -1.0

def test_correct_no_speedup_returns_one():
    reward = compute_reward(
        compiles=True,
        correct=True,
        speedup_vs_eager=1.0,
        speedup_vs_compile=0.9,
    )
    assert reward == 1.0

def test_beats_eager_returns_two():
    reward = compute_reward(
        compiles=True,
        correct=True,
        speedup_vs_eager=1.10,  # 10% faster than eager
        speedup_vs_compile=0.95,  # Slower than compile
    )
    assert reward == 2.0

def test_beats_both_returns_three():
    reward = compute_reward(
        compiles=True,
        correct=True,
        speedup_vs_eager=1.10,
        speedup_vs_compile=1.10,  # Beats both
    )
    assert reward == 3.0

def test_threshold_is_five_percent():
    # Exactly 5% should count
    reward = compute_reward(
        compiles=True,
        correct=True,
        speedup_vs_eager=1.05,
        speedup_vs_compile=1.05,
    )
    assert reward == 3.0
    
    # Just under 5% should not
    reward = compute_reward(
        compiles=True,
        correct=True,
        speedup_vs_eager=1.049,
        speedup_vs_compile=1.049,
    )
    assert reward == 1.0
```

**File:** `tests/test_compile.py`
```python
"""Tests for sm_80 compilation."""
import subprocess
import tempfile
import os

def test_nvcc_arch_flag():
    """Verify nvcc accepts -arch=sm_80."""
    with tempfile.NamedTemporaryFile(suffix=".cu", mode="w") as f:
        f.write('''
        extern "C" __global__ void test_kernel(float* x, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) x[i] *= 2.0f;
        }
        ''')
        f.flush()
        
        result = subprocess.run(
            ["nvcc", "-arch=sm_80", "-c", f.name, "-o", f.name + ".o"],
            capture_output=True,
            text=True,
        )
        
        # Should succeed (or skip if nvcc not available)
        if result.returncode != 0:
            if "nvcc: not found" in result.stderr:
                pytest.skip("nvcc not available")
            else:
                pytest.fail(f"nvcc failed: {result.stderr}")

def test_compile_timeout():
    """Verify compilation times out appropriately."""
    # ... test that long-running compilation is killed
    pass
```

---

## Part 5: Hackathon Execution Checklist

### 5.1 Pre-Hackathon (March 5-6)

- [ ] Download Qwen3-Coder-Next GPTQ-Int4 (~43GB) to local drive
- [ ] Download Qwen2.5-Coder-7B-Instruct (fallback)
- [ ] Download cudaLLM-8B for trajectory seeding
- [ ] Download CUDA-Agent-Ops-6K dataset
- [ ] Curate 200-problem subset
- [ ] Pre-compute eager/compile baselines (requires A100 access)
- [ ] Fix all P0 blockers
- [ ] Test environment end-to-end: reset() → step() → reward
- [ ] Test Qwen3-Coder-Next loading on H100 (GPTQ path)
- [ ] Test full pipeline: model load → generate → evaluate → gradient step

### 5.2 Day 1: March 7 (BUILD)

| Hour | Activity | Files |
|------|----------|-------|
| 0-1 | Claim GPU, load model, run Day-1 Verification Checklist | `training/model_loader.py` |
| 1-3 | Stage 1: GRPO warm-up (60-100 steps) | `training/stage1_warmup.py` |
| 3-4 | Stage 2: RFT (collect, filter, SFT) | `training/stage2_rft.py` |
| 4-7 | Stage 3: GRPO with curriculum (60 steps) | `training/stage3_grpo.py`, `training/curriculum.py` |
| 7-8 | Evaluate: compare base → warm-up → RFT → GRPO | `evaluation/eval_model.py` |
| 8-10 | Continue training if improving; run ablations | `evaluation/ablation.py` |

### 5.3 Day 2: March 8 (SHIP)

| Hour | Activity |
|------|----------|
| 0-2 | Final training + full evaluation suite |
| 2-4 | Build demo (Streamlit/Gradio): reward curve, kernel comparisons |
| 4-5 | Push model + environment to HuggingFace Hub |
| 5 | Pitch: 3 minutes |

---

## Part 6: Evidence References

| Claim | Source | Location |
|-------|--------|----------|
| Pure RL collapsed at step 17 | CUDA Agent paper | Section 3.3, page 6 |
| CUDA tokens = 0.01% of pretraining data | CUDA Agent paper | Section 3.3 |
| RFT ablation: faster-than-compile drops to 49.8% | CUDA Agent paper | Table 2, page 8 |
| Discrete rewards +36.4pp over continuous | CUDA Agent paper | Table 2 |
| Full system: 98.8% pass, 96.8% faster, 2.11× GM | CUDA Agent paper | Table 2 |
| 4-stage pipeline | CUDA Agent paper | Figure 3, page 5 |
| Reward function {-1,1,2,3} | CUDA Agent paper | Equation 1, page 5 |
| Anti-hacking measures | CUDA Agent paper | Section 3.2 |
| 6K operator dataset | CUDA Agent paper | Section 3.1 |
| 192 kernel files per GPU | DoubleGraph SKILLS.md | Section 1 |
| 3.6× average speedup over cuGraph | DoubleGraph announcement | March 2-3, 2026 |
| CachePool pattern (LRU, 8 entries) | DoubleGraph SKILLS.md | Section 3 |
| GRPO eliminates critic (~50% memory) | DeepSeek-R1 paper | Architecture section |
| cudaLLM-8B: 79.75% Bo1 Level-1 | ByteDance/cudaLLM-8B | Model card |

---

## Part 7: File Manifest Summary

### Files to FIX (existing but wrong)

| File | Tasks |
|------|-------|
| `training/stage3_grpo.py` | P0-1, P0-2, P1-8, P1-11 |
| `training/stage1_warmup.py` | P0-2, P1-7, P1-11 |
| `training/rft_filter.py` | P0-3, P0-4, P1-8, P1-15 |
| `verification/profile.py` | P0-5 |
| `skill_a100.md` | P1-13 |
| `openenv_env/kernel_forge_env.py` | P1-8 |

### Files to CREATE (missing)

| File | Purpose |
|------|---------|
| `training/model_loader.py` | P1-10: Fallback model selection |
| `training/stage1_warmup.py` | P1-7: GRPO warm-up (replace SFT) |
| `training/stage2_rft.py` | P1-15: Complete RFT with SFT |
| `training/stage3_grpo.py` | Rename from grpo_train.py + curriculum |
| `training/curriculum.py` | P1-9: Progressive difficulty |
| `datasets/download_ops6k.py` | P2-1: Dataset download |
| `datasets/curate_subset.py` | P2-1: 200-problem curation |
| `datasets/compute_baselines.py` | P2-1: Baseline timing |
| `evaluation/eval_model.py` | P2-2: Evaluation suite |
| `evaluation/ablation.py` | P2-2: H1/H2/H3 tests |
| `tests/test_env.py` | P2-3: Environment tests |
| `tests/test_reward.py` | P2-3: Reward tests |
| `tests/test_compile.py` | P2-3: Compilation tests |

---

**End of Architectural Task Document**
