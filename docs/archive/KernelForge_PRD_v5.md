# KernelForge-OpenEnv PRD v5.0

## Multi-Stage RL for GPU-Specific CUDA Kernel Generation

**Date:** March 3, 2026
**Hackathon:** OpenEnv Hackathon SF — March 7-8, 2026 (SHACK15, San Francisco)
**Organizers:** Cerebral Valley + PyTorch/Meta + SHACK15
**Prize Pool:** $100K+ cash | Teams up to 4

---

## 1. What We Are Building

### 1.1 The Goal

Build an RL post-training system that teaches Qwen3-Coder-Next (80B MoE, 3B active parameters) to generate optimized CUDA kernels targeting A100 (sm_80), using a multi-stage training pipeline adapted from ByteDance's CUDA Agent and architectural principles extracted from DoubleAI's WarpSpeed.

This is NOT about writing a single fast kernel. This IS about proving that a reproducible RL pipeline — environment, reward, curriculum, training stages — can measurably improve a model's ability to generate GPU-architecture-specific optimized kernels using an open-source framework anyone can run.

### 1.2 What Changed From v4.0

| Dimension | v4.0 | v5.0 (this document) |
|-----------|------|---------------------|
| **Model** | Qwen2.5-Coder-7B (dense, 7B) | Qwen3-Coder-Next (80B MoE, 3B active) |
| **Training approach** | "Pure RL" → corrected to SFT+GRPO | Multi-stage: Single-Turn RL → RFT → GRPO (CUDA Agent-validated) |
| **Why multi-stage** | Pragmatic choice | **Empirically required** — CUDA Agent collapsed at step 17 without warm-up |
| **Hardware** | A100 primary, A10G fallback | H100 80GB (GPTQ) or B200 192GB (bf16), cross-compile for A100 |
| **Kernel scope** | WCC only | Multi-algorithm: WCC + operator fusion + general CUDA ops |
| **Architecture awareness** | Generic optimizations | DoubleGraph-inspired per-GPU specialization patterns |
| **Reward design** | Simple 4-tier milestones | CUDA Agent's robust reward (validated via ablation: +36pp over continuous) |
| **Environment** | Single-algorithm OpenEnv | Multi-algorithm with curriculum progression and anti-hacking |
| **DoubleGraph integration** | Reference techniques only | Architectural patterns productized into training curriculum |

### 1.3 Why Multi-Stage Training Is Non-Negotiable

ByteDance's CUDA Agent paper (arXiv:2602.24286, Feb 27, 2026) provides the definitive evidence:

**The failure mode:** Their initial pure RL trial collapsed at training step 17. The root cause is a severe domain distribution mismatch — CUDA code represents less than 0.01% of pretraining data. CUDA tokens have probability ~10⁻⁹ in BF16 precision, causing the importance sampling ratio ρ_t(θ) = π_θ(a|s) / π_θ_old(a|s) to fluctuate wildly or explode when small numerical errors hit the precision floor.

**The ablation proof (Table 2):**

| Ablation | Pass Rate | Faster vs Compile | Speedup GM |
|----------|-----------|-------------------|------------|
| Full CUDA Agent | 98.8% | 96.8% | 2.11× |
| Without RFT (SFT warm-up) | 95.6% | 49.8% | 1.05× |
| Without Value Pretraining | 98.6% | 50.9% | 1.00× |
| Without Agent Loop | 77.1% | 14.1% | 0.69× |

Without RFT: training reward collapses, actor entropy spikes (policy becomes diffuse). Without Value Pretraining: critic fails to learn, trajectory lengths explode. Both ablations could only be reported at the "final validation step before training collapse."

**Our adaptation:** We use GRPO instead of PPO, which eliminates the critic model entirely. This means we skip Value Pretraining. But the RFT warm-up stage is mandatory — it provides the behavioral prior that prevents policy collapse. Our pipeline has 3 stages instead of 4.

### 1.4 Why This Wins the Hackathon

1. **Uses OpenEnv** — the hackathon's framework. We build reusable, gym-compatible CUDA environments.
2. **Demonstrates RL post-training** — the hackathon's thesis. Not inference-time search, actual policy improvement with a validated multi-stage methodology.
3. **Grounded in published evidence** — every design choice traces to an ablation in the CUDA Agent paper or an architectural pattern from DoubleGraph.
4. **Multi-algorithm scope** — WCC, operator fusion, general operators. Not a single-kernel demo.
5. **Productizable** — the per-GPU specialization pipeline (from DoubleGraph) combined with RL training (from CUDA Agent) is a complete system, not a research toy.

---

## 2. DoubleGraph Analysis: What WarpSpeed Did, How, and Why

### 2.1 The Core Insight

DoubleAI's WarpSpeed replaced NVIDIA cuGraph's architecture-generic kernels with per-GPU-architecture optimized implementations. The system generates 192 CUDA kernel files per target GPU, each with fundamentally different algorithm strategies tuned for specific hardware. Result: 3.6× average speedup over expert-tuned code, with 18% of algorithms achieving 10-100× speedup.

The key insight is not that they wrote faster kernels. It's that **they built an architecture for systematically producing GPU-specific kernels** — and that architecture is what we can train an RL agent to operate within.

### 2.2 How They Structured It (5 Layers)

**Layer 1 — Universal Graph Abstraction (compact_graph_t):**
A stripped-down CSR/CSC type that bridges cuGraph's complex template world to the optimized kernel world. Every kernel takes the same input type. This eliminates template metaprogramming from hot paths and gives the optimizer a stable interface.

**Why this matters for us:** Our RL agent needs a fixed kernel interface contract. DoubleGraph proved that a universal abstraction (compact_graph_t) is sufficient for all graph algorithms. We adapt this: every kernel the agent generates must implement a fixed `extern "C"` signature for its algorithm type.

**Layer 2 — GPU Resource Management (CachePool):**
Thread-local LRU cache of GPU resources (8 entries per thread). Repeated algorithm calls reuse previously allocated buffers instead of cycling through cudaMalloc/cudaFree. Uses RAII via Cacheable subclasses.

**Why this matters for us:** Our evaluation environment runs hundreds of kernel evaluations per GRPO step. Naive cudaMalloc/cudaFree per evaluation wastes time. We implement a CachePool-style resource manager in the OpenEnv environment to keep test graph data resident on GPU across evaluations.

**Layer 3 — 4-Way Dispatch:**
Every algorithm has 4 variants based on two binary properties: has segment_offsets (degree-based vertex partitioning) × has edge_mask (bitmask for subgraph operations). Each variant is a separate function with a separate .cu file — not runtime branching but compile-time specialization.

**Why this matters for us:** This is our curriculum structure. An RL agent doesn't need to learn all 4 variants at once. We start with the base variant, then promote to segmented (teaches degree-aware scheduling), then masked (teaches edge filtering), then seg+mask. Each variant is a harder optimization target.

**Layer 4 — Per-GPU Kernel Implementations:**
The actual optimization work. Each GPU gets 192 .cu files that differ fundamentally between architectures. A100 BFS uses direction-optimizing search with queue↔bitmap frontier conversion. L4 BFS uses bitmap-only with CUB. A10G uses 2-level batched top-down with L2 pinning. Same algorithm, completely different implementations.

**Why this matters for us:** This is the evidence that GPU-specific optimization is not just parameter tuning — it requires different algorithmic strategies. An RL agent needs to discover these strategy-level differences, not just tune block sizes.

**Layer 5 — Build System (compile-time GPU selection):**
GPU target selected at build time via `-DTARGET_GPU=A100`. Per-kernel .cu.flags sidecars control compilation options (--use_fast_math, --rdc=true, --maxrregcount=48). No runtime dispatch.

**Why this matters for us:** We cross-compile for A100 using `-arch=sm_80` on whatever GPU the hackathon provides. The .cu.flags pattern teaches us that per-kernel compilation flags are an optimization dimension the agent should learn to specify.

### 2.3 A100-Specific Techniques (What WarpSpeed Did for SM80)

These are the concrete patterns the RL agent should learn to discover:

**Direction-Optimizing BFS with Cost Modeling:**
Two phases — top-down (queue, sparse frontier) and bottom-up (bitmap, dense frontier) — with switching based on frontier size vs graph size. A100's 40MB L2 makes bitmap-based bottom-up cheap because large bitmaps fit in cache. Switching thresholds: TD→BU at frontier_size > N/20, BU→TD at frontier_size < N/200.

**Queue↔Bitmap Frontier Conversion:**
Maintain both representations. Use warp-level `__ballot_sync` to parallelize bit scanning during conversion. This is A100-specific because the large L2 can hold both representations simultaneously.

**Hand-Rolled Warp Primitives (No CUB):**
A100 kernels use custom `__shfl_down_sync` reductions instead of CUB library calls. This gives tighter register control and eliminates library call overhead. L4 uses CUB instead — the optimal choice depends on the architecture's register file and scheduler.

**3-Tier Louvain Dispatch:**
Community detection dispatches based on graph coarsening level: serial (N ≤ 200), thread-per-vertex with register hash table (avg_degree < 8), warp-per-vertex with shared-memory hash table (avg_degree ≥ 8). Hash table uses linear probing with `atomicCAS`.

**Cooperative Kernels:**
For low-degree masked graphs, persistent thread blocks with grid-wide `cg::this_grid().sync()` between BFS levels. Eliminates kernel launch overhead entirely. Requires `--rdc=true` and `CUDA_SEPARABLE_COMPILATION=ON`.

**SpMV-Based PageRank:**
Combines cuSPARSE for the matrix-vector product with custom kernels for dangling node sum (atomic reduction) and fused update+convergence check (single pass over rank vector).

### 2.4 The Meta-Pattern: What's Transferable

DoubleGraph's most valuable contribution is not any individual kernel — it's the **process**:

1. **Characterize the hardware** — L2 size, SMEM capacity, register file, warp scheduler
2. **Profile generic kernels** — Run the original on target hardware, find bottlenecks
3. **Choose algorithm strategy** — Pick representations (bitmap vs queue vs hybrid) based on memory hierarchy
4. **Tune kernel parameters** — Block size, register count, shared memory allocation
5. **Validate with per-kernel flags** — Use .cu.flags for compilation control without build system changes

This 5-step process is exactly what we want the RL agent to internalize. The SKILL.md given to the agent during training encodes steps 1-2 as context. The agent must learn steps 3-5 through RL.

---

## 3. Productization: How We Combine CUDA Agent + DoubleGraph

### 3.1 The Synthesis

CUDA Agent proved that RL can train a model to write optimized CUDA kernels. DoubleGraph proved that GPU-specific kernel specialization yields massive speedups. We combine them:

**CUDA Agent provides the training methodology:**
- Multi-stage pipeline (warm-up → RFT → RL)
- Skill-integrated agent loop (SKILL.md protocol)
- Robust milestone-based rewards (not continuous speedup)
- Anti-reward-hacking measures

**DoubleGraph provides the optimization target space:**
- Per-GPU specialization as curriculum (same algorithm, different strategies per arch)
- 4-way dispatch as progressive difficulty levels
- Concrete A100-specific technique library for SKILL.md
- .cu.flags as learnable compilation parameters
- CachePool for efficient environment evaluation

**Our contribution is the bridge:**

```
DoubleGraph Architecture Patterns     CUDA Agent Training Methodology
    (what to optimize)          ×     (how to train the optimizer)
           ↓                                    ↓
    ┌──────────────────────────────────────────────────┐
    │           KernelForge v5.0 Pipeline              │
    │                                                  │
    │  SKILL.md encodes DoubleGraph's per-GPU          │
    │  technique library as agent context              │
    │                                                  │
    │  Environment implements DoubleGraph's             │
    │  compile→verify→profile loop with                │
    │  CUDA Agent's anti-hacking measures              │
    │                                                  │
    │  Curriculum uses DoubleGraph's 4-way             │
    │  dispatch as progressive difficulty              │
    │                                                  │
    │  Multi-stage training follows CUDA Agent's       │
    │  validated pipeline (warm-up → RFT → GRPO)       │
    │                                                  │
    │  Reward uses CUDA Agent's discrete milestones    │
    │  with DoubleGraph's relative speedup metric      │
    └──────────────────────────────────────────────────┘
```

### 3.2 What We Productize Beyond the Hackathon

The hackathon deliverable is a proof-of-concept. The productizable system is:

**A GPU-Specific Kernel Generation Pipeline** that takes:
- Input: A PyTorch operator + target GPU architecture
- Process: RL-trained model generates architecture-specific CUDA kernel
- Output: Optimized .cu file + .cu.flags sidecar + integration binding

This is DoubleGraph's 192-kernel-per-GPU approach, but automated via RL instead of manual engineering. The value proposition: DoubleGraph covers 8 algorithm categories with 192 kernels per GPU. Scaling to new GPUs (H100, B200, B300) requires re-implementing all 192 kernels per architecture. An RL-trained model that understands GPU-specific optimization could generate these variants automatically.

### 3.3 Efficiency Gains From the Combination

| Without DoubleGraph Patterns | With DoubleGraph Patterns |
|------------------------------|--------------------------|
| Agent explores random optimization space | Agent explores structured optimization space (4-way dispatch progression) |
| Reward only measures "faster/slower" | Reward measures specific milestones aligned with known optimization tiers |
| No curriculum — all kernels equally hard | Curriculum: base → segmented → masked → seg+mask |
| Agent must discover GPU-specific techniques from scratch | SKILL.md encodes known A100 techniques; agent refines and combines them |
| Evaluation allocates GPU resources per-call | CachePool keeps resources resident across evaluations |
| Compilation flags are fixed | Agent learns to specify per-kernel .cu.flags |

---

## 4. Hardware Configuration

### 4.1 Why H100 or B200 (Not A100 for Training)

Qwen3-Coder-Next is an 80B MoE model. Even with aggressive quantization, it needs substantial VRAM for training:

| Config | Weights | LoRA | Optimizer | Activations | KV Cache | Overhead | Total | Fits? |
|--------|---------|------|-----------|-------------|----------|----------|-------|-------|
| H100 80GB + 4-bit GPTQ + QLoRA | ~43 GB | 0.05 GB | 0.3 GB | 3 GB | 1.5 GB | 5 GB | **~53 GB** | Yes (27GB headroom) |
| H100 80GB + FP8 | ~80 GB | — | — | — | — | — | **~80 GB** | No (zero headroom) |
| B200 192GB + bf16 + LoRA | ~160 GB | 0.2 GB | 1.6 GB | 4 GB | 2 GB | 3 GB | **~171 GB** | Yes (21GB headroom) |

**Primary path: H100 80GB + 4-bit GPTQ + QLoRA**
- GPTQModel supports MoE with Moe.Routing
- Community GPTQ-INT4 exists: `dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16`
- **Blocker:** Unsloth does not support GPTQ-based QLoRA — must use HuggingFace PEFT + Transformers
- Sacrifices Unsloth speed optimizations but is functional

**Fallback path: B200 192GB + bf16 + LoRA**
- Unsloth confirms Qwen3-Coder-Next fits in bf16 LoRA on single B200
- Unsloth fully supports B200 (sm_100/Blackwell)
- LoRA targets attention + shared expert layers only (all 512 experts = 207GB, does not fit)

### 4.2 Cross-Compilation for A100 (sm_80)

We train on H100/B200 but generate kernels targeting A100:

```bash
# Cross-compile on H100 for A100 target
nvcc -arch=sm_80 -O3 -use_fast_math kernel.cu -o kernel.so --shared -Xcompiler -fPIC
```

Binary includes sm_80 cubin. When executed on H100, CUDA driver JIT-compiles the sm_80 PTX to sm_90. The kernel runs correctly but performance characteristics differ (H100 has 132 SMs vs 108, 50MB L2 vs 40MB, ~3.35 TB/s vs ~2 TB/s bandwidth).

**For reward function: use relative speedup** (optimized/naive baseline compiled for the same target) rather than absolute timing. Both the agent's kernel and the baseline are cross-compiled for sm_80, so the relative comparison is valid even when running on H100.

### 4.3 What Runs on the Single GPU

```
┌─────────────────────────────────────────────────────────┐
│                Single H100 80GB GPU                      │
│                                                          │
│  ┌───────────────────────────────┐                       │
│  │ Qwen3-Coder-Next (GPTQ 4-bit)│   ~53 GB              │
│  │ + QLoRA adapters              │                       │
│  └───────────────┬───────────────┘                       │
│                  │                                        │
│                  │ Generate CUDA code (single-turn)       │
│                  ▼                                        │
│  ┌───────────────────────────────┐                       │
│  │ nvcc cross-compile sm_80     │   CPU-side, ~3-5s      │
│  │ + .cu.flags applied          │                        │
│  └───────────────┬───────────────┘                       │
│                  │                                        │
│                  │ Load .so, execute on GPU                │
│                  ▼                                        │
│  ┌───────────────────────────────┐                       │
│  │ CachePool evaluation         │   ~1-2 GB GPU memory   │
│  │  • 5 test inputs             │   Resources stay       │
│  │  • PAC verification          │   resident via LRU     │
│  │  • Profiling (50 warm + 30)  │                        │
│  └───────────────┬───────────────┘                       │
│                  │                                        │
│                  │ Return discrete reward                  │
│                  ▼                                        │
│  ┌───────────────────────────────┐                       │
│  │ GRPO gradient step           │   27 GB headroom       │
│  │ (no critic model needed)     │                        │
│  └───────────────────────────────┘                       │
│                                                          │
│  Total per step: ~60-90 seconds                          │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Training Pipeline: The Validated 3-Stage Approach

### 5.1 Overview

```
CUDA Agent (230B, 128 GPUs)        KernelForge v5.0 (80B MoE, 1 GPU)
───────────────────────────        ─────────────────────────────────
Stage 1: Single-turn PPO           Stage 1: Single-turn GRPO (2 hrs)
         warm-up                            warm-up on easy kernels
Stage 2: RFT on agent              Stage 2: RFT on filtered          
         trajectories                       trajectories (30 min)
Stage 3: Critic Value              SKIPPED (GRPO has no critic)
         Pretraining
Stage 4: Multi-turn agentic        Stage 3: Single-turn GRPO with
         PPO (150 steps)                    curriculum (2-3 hrs)
```

**Why single-turn, not multi-turn agentic:** ByteDance's multi-turn agent loop used 128K context and up to 200 interaction turns. On a single GPU with a 80B model, we cannot afford 128K context windows. We run single-turn GRPO where the agent generates one kernel per prompt, receives a reward, and the policy updates. This is closer to their "single-turn warm-up" stage, which alone achieved meaningful improvement over the base model.

### 5.2 Stage 1: Single-Turn GRPO Warm-Up (2 hours)

**Goal:** Bootstrap the model's CUDA kernel generation from "knows CUDA syntax" to "can write kernels that compile and pass correctness."

**Why this works:** Qwen3-Coder-Next has CUDA in its pretraining data (it's a dedicated coder model). The warm-up stage doesn't teach CUDA from scratch — it shapes the model's existing knowledge toward the specific kernel generation task and format.

**Training data:** CUDA-Agent-Ops-6K dataset (6,000 operators, Apache 2.0, published by ByteDance at `BytedTsinghua-SIA/CUDA-Agent-Ops-6K`). Each operator is a PyTorch nn.Module with `forward()`, `get_inputs()`, and `get_init_inputs()`. We use a 200-problem subset, starting with the simplest operators (single-op from torch library, execution time near 1ms).

**Curriculum — start trivially easy:**
- First 30 steps: Single torch operators (relu, sigmoid, matmul, conv2d)
- Steps 30-60: 2-operator compositions (matmul+relu, conv2d+bias+relu)
- Steps 60-100: 3+ operator fusions (the core CUDA Agent training distribution)

**Online prompt filtering (PRIME technique):** Maintain prompts at 20-80% success rate. If a prompt is too hard (>80% failure), demote it. If too easy (>80% success), promote to harder variants.

```python
from trl import GRPOConfig, GRPOTrainer

warmup_config = GRPOConfig(
    learning_rate=3e-6,
    num_generations=8,              # 8 kernels per prompt (DeepSeek-R1 uses 16)
    max_prompt_length=1024,
    max_completion_length=4096,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=100,
    temperature=0.9,                # Higher early for exploration
    beta=0.0,                       # No KL penalty (recent research)
    bf16=True,
    optim="paged_adamw_8bit",
    output_dir="outputs/stage1-warmup",
    logging_steps=1,
)

def warmup_reward(completions, **kwargs) -> list[float]:
    """Reward for single-turn kernel generation."""
    env = KernelForgeEnv()
    rewards = []
    for completion in completions:
        code = extract_cuda_code(completion)
        if not code:
            rewards.append(-1.0)
            continue
        result = env.evaluate_kernel(code)
        rewards.append(result.reward)
    return rewards

warmup_trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[warmup_reward],
    args=warmup_config,
    train_dataset=easy_prompt_dataset,
)
warmup_trainer.train()
```

**Expected outcome:** Model goes from ~50% compilation rate to ~85%+ compilation rate. Average reward moves from ~0.0 (mix of -1 failures and +1 correct) to ~0.8-1.2 (mostly correct, occasionally fast).

### 5.3 Stage 2: Rejection Fine-Tuning (30 minutes)

**Goal:** Filter the warm-up model's trajectories for quality, then supervised fine-tune to create a strong behavioral prior for the final RL stage.

**Why RFT is mandatory (CUDA Agent Figure 4):** Without RFT, training reward collapses within ~15 steps. Actor entropy spikes as the policy becomes diffuse and unstructured. RFT constrains entropy growth during subsequent RL.

```python
# Collect trajectories from Stage 1 model
trajectories = []
for prompt in curriculum_prompts:  # 200 diverse operators
    for _ in range(4):  # 4 generations per prompt
        completion = generate(model, tokenizer, prompt)
        code = extract_cuda_code(completion)
        result = env.evaluate_kernel(code)
        trajectories.append({
            "prompt": prompt,
            "completion": completion,
            "reward": result.reward,
            "speedup": result.info.get("speedup", 0),
        })

# Filter: keep reward >= 1.0 (correct) AND well-formatted
good_trajectories = [
    t for t in trajectories
    if t["reward"] >= 1.0
    and not has_invalid_patterns(t["completion"])  # No hallucinated tool calls
]

# SFT on filtered data
from trl import SFTConfig, SFTTrainer

rft_config = SFTConfig(
    output_dir="outputs/stage2-rft",
    max_seq_length=4096,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=100,
    learning_rate=2e-5,
    bf16=True,
    optim="paged_adamw_8bit",
)

rft_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=Dataset.from_list(good_trajectories),
    args=rft_config,
)
rft_trainer.train()
```

**Collection time:** 200 prompts × 4 generations × ~15s = ~200 minutes. **Optimization:** Parallelize generation if multi-GPU available, or reduce to 50 prompts × 4 = ~50 minutes.

**RFT training time:** 100 steps × ~3s/step = ~5 minutes.

### 5.4 Stage 3: GRPO Training with Curriculum (2-3 hours)

**Goal:** RL fine-tuning on progressively harder kernels. This is where the model should discover optimization strategies beyond what was in the RFT data.

**Curriculum progression (inspired by DoubleGraph's 4-way dispatch):**

| Phase | Steps | Target | Example Optimizations |
|-------|-------|--------|----------------------|
| Phase A: Base | 0-30 | Single operators, no special dispatch | Memory coalescing, shared memory tiling |
| Phase B: Fusion | 30-60 | 2-3 operator compositions | Kernel fusion, eliminate intermediate tensors |
| Phase C: Architecture | 60-90 | A100-specific optimizations | L2 pinning, vectorized float4, warp primitives |
| Phase D: Advanced | 90-120 | Complex operators | Algebraic simplification, library integration (cuBLAS/cuDNN) |

```python
grpo_config = GRPOConfig(
    learning_rate=5e-6,
    num_generations=8,
    max_prompt_length=1024,
    max_completion_length=4096,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=120,
    temperature=0.7,                # Lower than warmup — exploit more
    beta=0.0,
    bf16=True,
    optim="paged_adamw_8bit",
    output_dir="outputs/stage3-grpo",
    logging_steps=1,
)

grpo_trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[cuda_reward_with_curriculum],
    args=grpo_config,
    train_dataset=curriculum_dataset,  # Updates per phase
)
grpo_trainer.train()
```

**Time estimate per step:** 8 generations × ~15s inference + 8 × ~10s evaluation + ~5s gradient = ~205s/step. 120 steps = ~410 minutes (~7 hours). This is too long for a hackathon. **Mitigation:** reduce to `num_generations=4` and `max_steps=60`, bringing total to ~100 minutes.

---

## 6. The OpenEnv Environment (v5.0)

### 6.1 Architecture: Multi-Algorithm with CachePool

The v5.0 environment supports multiple kernel types, implements DoubleGraph-inspired resource caching, and includes CUDA Agent's anti-hacking measures.

```python
"""
KernelForge OpenEnv Environment v5.0

Multi-algorithm CUDA kernel optimization environment with:
- DoubleGraph-inspired CachePool for GPU resource reuse
- CUDA Agent-validated reward milestones
- Anti-reward-hacking measures (protected scripts, randomized inputs)
- Cross-compilation for A100 (sm_80) on any GPU

Install: pip install "openenv-core[core]>=0.2.1"
"""
import subprocess
import tempfile
import os
import ctypes
import hashlib
import numpy as np
from openenv.core.env_server import Environment, create_fastapi_app
from openenv.core.types import StepResult


class GPUCachePool:
    """
    DoubleGraph-inspired GPU resource cache.
    Keeps test data resident on GPU across evaluations.
    Thread-local, LRU eviction, max 8 entries.
    """
    _MAX = 8

    def __init__(self):
        self._cache = {}
        self._order = []

    def get_or_create(self, key, create_fn):
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        if len(self._cache) >= self._MAX:
            evict_key = self._order.pop(0)
            del self._cache[evict_key]
        val = create_fn()
        self._cache[key] = val
        self._order.append(key)
        return val


class KernelForgeEnv(Environment):
    """
    Multi-algorithm CUDA kernel optimization environment.
    """

    # --- Configuration ---
    ARCH_FLAG = "-arch=sm_80"         # Target: A100
    NVCC_FLAGS = ["-O3", "-use_fast_math", "--shared",
                  "-Xcompiler", "-fPIC"]
    MAX_COMPILE_SECONDS = 30
    WARMUP_ITERS = 50
    BENCHMARK_RUNS = 30
    VERIFY_INPUTS = 5                 # 5 randomized inputs per evaluation

    def __init__(self):
        self.cache_pool = GPUCachePool()
        self.current_problem = None
        self.baseline_ms = None

    def reset(self, problem=None):
        """
        Reset with a specific problem (PyTorch operator).
        Problem format matches CUDA-Agent-Ops-6K.
        """
        self.current_problem = problem or self._sample_problem()
        self.baseline_ms = self._profile_baseline()
        return {
            "observation": self._build_prompt(),
            "baseline_ms": self.baseline_ms,
            "target_arch": self.ARCH_FLAG,
        }

    def step(self, action: str) -> StepResult:
        """Evaluate a CUDA kernel submission."""
        return self.evaluate_kernel(action)

    def evaluate_kernel(self, cuda_code: str) -> StepResult:
        """
        Full evaluation pipeline:
        1. Compile with nvcc (cross-compile for sm_80)
        2. Verify entry point exists
        3. Run correctness checks on 5 randomized inputs
        4. Benchmark against baseline
        5. Compute discrete milestone reward
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "kernel.cu")
            lib = os.path.join(tmpdir, "kernel.so")

            # Apply .cu.flags if specified in the submission
            flags = self._extract_cu_flags(cuda_code)
            nvcc_flags = self.NVCC_FLAGS + flags

            with open(src, "w") as f:
                f.write(cuda_code)

            # Step 1: Compile
            cmd = ["nvcc", self.ARCH_FLAG] + nvcc_flags + [src, "-o", lib]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=self.MAX_COMPILE_SECONDS,
                )
                if proc.returncode != 0:
                    return self._result(-1.0,
                        f"COMPILE ERROR:\n{proc.stderr[:1500]}")
            except subprocess.TimeoutExpired:
                return self._result(-1.0, "COMPILE TIMEOUT (30s)")

            # Step 2: Load and verify entry point
            try:
                so = ctypes.CDLL(lib)
            except OSError as e:
                return self._result(-1.0, f"LOAD ERROR: {e}")

            entry = self.current_problem.get("entry_point", "optimized_forward")
            if not hasattr(so, entry):
                return self._result(-1.0,
                    f"MISSING: extern \"C\" void {entry}(...)")

            # Step 3: Correctness on 5 randomized inputs
            # (anti-hacking: different shapes each time)
            for i in range(self.VERIFY_INPUTS):
                inputs = self._generate_random_inputs(seed=i)
                try:
                    kernel_out = self._run_kernel(so, entry, inputs)
                    ref_out = self._run_reference(inputs)
                except Exception as e:
                    return self._result(-1.0,
                        f"RUNTIME ERROR (input {i}): {str(e)[:500]}")

                if not self._check_correctness(kernel_out, ref_out):
                    return self._result(-1.0,
                        f"WRONG OUTPUT (input {i}): "
                        f"max_diff={np.max(np.abs(kernel_out - ref_out)):.6f}")

            # Step 4: Benchmark
            runtime_ms = self._benchmark(so, entry)
            speedup_vs_baseline = self.baseline_ms / runtime_ms

            # Step 5: CUDA Agent discrete reward
            if speedup_vs_baseline > 1.50:
                reward = 3.0    # Beats baseline by >50%
            elif speedup_vs_baseline > 1.05:
                reward = 2.0    # Meaningful speedup (>5%)
            else:
                reward = 1.0    # Correct but not faster

            obs = (
                f"PASS | Runtime: {runtime_ms:.3f}ms | "
                f"Baseline: {self.baseline_ms:.3f}ms | "
                f"Speedup: {speedup_vs_baseline:.2f}x | "
                f"Reward: {reward}"
            )

            return self._result(reward, obs, info={
                "speedup": speedup_vs_baseline,
                "runtime_ms": runtime_ms,
            })

    def _extract_cu_flags(self, code):
        """
        DoubleGraph-inspired: extract per-kernel .cu.flags from
        a special comment in the CUDA code.
        Format: // CU_FLAGS: --use_fast_math --maxrregcount=48
        """
        flags = []
        for line in code.split('\n'):
            if line.strip().startswith('// CU_FLAGS:'):
                flag_str = line.strip().replace('// CU_FLAGS:', '').strip()
                # Whitelist safe flags only (anti-hacking)
                allowed = {'--use_fast_math', '--extra-device-vectorization',
                          '--rdc=true'}
                allowed_prefixes = ('--maxrregcount=',)
                for f in flag_str.split():
                    if f in allowed or any(f.startswith(p) for p in allowed_prefixes):
                        flags.append(f)
        return flags

    # ... (benchmark, reference, correctness methods similar to v4.0)

    @property
    def state(self):
        return {"problem": self.current_problem}


app = create_fastapi_app(KernelForgeEnv)
```

### 6.2 Anti-Reward-Hacking Measures (From CUDA Agent Section 3.2)

CUDA Agent documented specific exploits they had to prevent. We implement all of them:

| Measure | Implementation |
|---------|---------------|
| **Protected evaluation scripts** | Verification and profiling run in a subprocess with read-only access to the evaluation code. Agent cannot modify scoring logic. |
| **Forbid fallback calls** | Generated kernels cannot call `torch.nn.functional` or `torch.compile`. We scan the compiled .so for these symbols and reject. |
| **Randomized inputs** | Each evaluation uses 5 randomly-shaped inputs (not fixed shapes). Agent cannot memorize outputs. |
| **Synchronized profiling** | Proper `cudaDeviceSynchronize()` before and after benchmarking. Warm-up iterations prevent CUDA graph exploitation. |
| **No web search** | Agent has no external information retrieval — solutions come from model weights + SKILL.md only. |
| **Whitelisted .cu.flags** | Only safe compilation flags accepted (--use_fast_math, --maxrregcount). No arbitrary flags. |

### 6.3 SKILL.md v5.0 (Encoding DoubleGraph Techniques)

The SKILL.md is adapted from CUDA Agent's original (Appendix B.2) but enriched with A100-specific techniques from DoubleGraph:

```markdown
# KernelForge SKILL.md v5.0 — A100-Optimized CUDA Kernel Generation

## Target Hardware
- GPU: NVIDIA A100 (Ampere, sm_80)
- SMs: 108 | L2 Cache: 40 MB | Shared Memory/SM: 164 KB
- HBM2e: 2.0 TB/s | Registers/SM: 65,536
- nvcc: -arch=sm_80 -O3 -use_fast_math

## Your Task
Write an optimized CUDA C++ extension that accelerates the given
PyTorch operator. Your code will be cross-compiled for A100.

## Kernel Interface
Your code MUST define:
    extern "C" void optimized_forward(/* problem-specific args */);

## Per-Kernel Compilation Flags
You may specify compilation flags in a comment:
    // CU_FLAGS: --use_fast_math --maxrregcount=48
Allowed flags: --use_fast_math, --maxrregcount=N, --rdc=true,
               --extra-device-vectorization

## A100 Optimization Techniques (Priority Order)

### Priority 1: Algorithmic (>50% impact)
- **Algebraic simplification**: Reduce computation before optimizing.
  Example: diag(A) × B = row-wise scaling (O(N²M) → O(NM))
- **Kernel fusion**: Merge sequential ops into one kernel.
  Eliminates intermediate tensor materialization.
- **Operator reduction**: Replace generic ops with specialized kernels
  matching the mathematical structure.

### Priority 2: A100 Memory Hierarchy (20-50% impact)
- **L2 cache pinning**: A100 has 40MB L2. Pin frequently accessed
  data (parent arrays, frontier bitmaps, weight matrices):
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 30*1024*1024);
    // Set stream attribute for persistent access
- **Vectorized loads**: Use float4 for 16-byte aligned accesses.
  4x memory transaction efficiency.
- **Memory coalescing**: Consecutive threads access consecutive addresses.
- **Shared memory tiling**: Cache frequently accessed data in 164KB SMEM.
  Pad arrays: float tile[32][33] to avoid 32-bank conflicts.

### Priority 3: A100 Compute (10-20% impact)
- **Warp primitives**: __shfl_sync, __ballot_sync for intra-warp ops.
  Hand-rolled reductions give tighter register control than CUB.
- **Occupancy tuning**: __launch_bounds__(256, 4) for register control.
  Use cudaOccupancyMaxActiveBlocksPerMultiprocessor.
- **Cooperative kernels**: For iterative algorithms, use
  cg::this_grid().sync() to avoid kernel launch overhead.
  Requires: // CU_FLAGS: --rdc=true
- **TF32 for matmul**: torch.backends.cuda.matmul.allow_tf32 = True
  leverages A100 Tensor Cores for ~3x GEMM throughput.

### Priority 4: Library Integration
- **cuBLAS for GEMM**: When the operator involves matrix multiply,
  use cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32.
- **cuDNN for Conv**: Use cudnnConvolutionBiasActivationForward
  to fuse conv + bias + activation in one call.
- **cuSPARSE for SpMV**: For sparse operations, cusparseSpMV
  with CSR format.

## Rules
- Do NOT modify verification or profiling scripts
- Do NOT call torch.nn.functional or torch.compile in generated kernels
- Do NOT hardcode outputs for specific inputs
- Your kernel MUST work for any valid input of the specified type
```

---

## 7. Reward Design

### 7.1 CUDA Agent's Validated Scheme

CUDA Agent's ablation (Table 2) proved that discrete milestone rewards significantly outperform continuous speedup rewards. Continuous rewards suffer from noise (thermal throttling, OS interrupts) that creates outlier advantage estimates and biases the policy toward easy kernels.

Our reward function:

```
r = -1.0    if correctness check fails
r = +1.0    if correct, speedup ≤ 1.05× vs baseline
r = +2.0    if correct, speedup > 1.05× vs torch.eager equivalent
r = +3.0    if correct, speedup > 1.05× vs both eager AND torch.compile
```

This maps directly to CUDA Agent's {-1, 1, 2, 3} scheme where:
- r=1: correct but not faster
- r=2: beats eager execution (basic optimization)
- r=3: beats torch.compile (strong optimization)

### 7.2 Baseline Profiling

For each operator in the training set, we pre-compute:
1. **torch.eager runtime**: Run the original PyTorch model with `torch.no_grad()`, median of 30 runs
2. **torch.compile runtime**: Run `torch.compile(model)`, median of 30 runs after warm-up

Both baselines are cross-compiled/run on the same GPU as the agent's kernels, ensuring relative speedup comparisons are valid.

---

## 8. Model Configuration

### 8.1 Primary: H100 80GB + GPTQ-Int4

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

# Load GPTQ-quantized model
model = AutoModelForCausalLM.from_pretrained(
    "dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(
    "dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16"
)

# QLoRA on attention + shared expert
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        # Shared expert only (not all 512 routed experts)
        "shared_expert.gate_proj",
        "shared_expert.up_proj",
        "shared_expert.down_proj",
    ],
    lora_dropout=0,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

### 8.2 Fallback: B200 192GB + bf16 + Unsloth

```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/Qwen3-Coder-Next-80B-A3B-Instruct",
    max_seq_length=8192,
    load_in_4bit=False,  # bf16, not quantized
    dtype=torch.bfloat16,
)

model = FastModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        # Attention-only LoRA (shared expert too large on bf16)
    ],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
```

### 8.3 Day-1 Verification Checklist

Before committing to either path, test on hackathon hardware:

1. **Does the quantized model load?** Try loading `dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16` and check peak VRAM.
2. **Does QLoRA attach?** Verify PEFT LoRA modules can be added to MoE architecture.
3. **Can it generate CUDA?** Prompt with a simple operator and check if output contains valid CUDA syntax.
4. **Does gradient flow?** Run 1 training step and verify loss changes.
5. **Baseline CUDA quality?** Generate 10 kernels for easy operators, check compilation rate. If >70% compile without any training, the model has strong CUDA priors and the warm-up stage can be shorter.

---

## 9. Hackathon Execution Plan

### 9.1 Before the Hackathon (Now Through March 6)

**Day -3 to -2: Data and Infrastructure**

1. Download CUDA-Agent-Ops-6K from HuggingFace
2. Curate 200-problem subset: 50 easy (single-op), 75 medium (2-op fusion), 75 hard (3+ ops)
3. For each: pre-compute torch.eager and torch.compile baselines on cloud GPU
4. Test the OpenEnv environment: reset() → step() → reward
5. Download and verify GPTQ model loads (may need hours over slow networks)

**Day -1: Pipeline Verification**

6. Run 3-step GRPO warm-up to verify gradient flow
7. Run 1 trajectory collection + RFT filter to verify Stage 2
8. Time everything. Know exact seconds per training step.
9. Prepare fallback: if GPTQ model fails, have Qwen2.5-Coder-7B ready
10. Pre-generate 20 "seed trajectories" using cudaLLM-8B for Expert Iteration fallback

### 9.2 Day 1: March 7 (BUILD)

**Hour 0-1: Hardware Verification**
- Claim GPU, verify type (H100? B200? A100?)
- Load model, run Day-1 Verification Checklist (Section 8.3)
- If GPTQ fails: fall back to Qwen2.5-Coder-7B + Unsloth (v4.0 path)
- Deploy OpenEnv environment, verify full loop

**Hour 1-3: Stage 1 — Single-Turn GRPO Warm-Up**
- Load easy operator subset (50 single-op problems)
- Run GRPO: 60 steps, num_generations=4, temperature=0.9
- Monitor: compilation rate should climb from ~50% to ~85%
- If stuck at <30% compilation after 20 steps: model can't generate CUDA, abort to Qwen2.5-Coder-7B

**Hour 3-4: Stage 2 — RFT**
- Generate 50 prompts × 4 completions from warm-up model
- Filter: keep reward ≥ 1.0 (correct kernels)
- SFT on filtered data: 100 steps (~5 min)

**Hour 4-7: Stage 3 — GRPO with Curriculum**
- Start with medium operators (2-op fusion), run 30 steps
- Promote to hard operators if >50% achieve reward ≥ 2.0
- Monitor reward distribution every 10 steps
- If training collapses (reward drops to -1.0): increase temperature, reduce learning rate

**Hour 7-8: Evaluate and Decide**
- Run evaluation: 50 prompts, compare base → warm-up → RFT → GRPO
- If reward curve shows improvement: continue to Day 2
- If flat: analyze failure mode, prepare "interesting negative result" framing

### 9.3 Day 2: March 8 (SHIP)

**Hour 0-2: Final Training + Extended Evaluation**
- Continue GRPO on hardest operators if still improving
- Run full evaluation suite: 100 prompts across all difficulty levels
- Compute ablation: SFT-only vs SFT+GRPO (A1), with SKILL.md vs without (A2)

**Hour 2-4: Build Demo**
- Streamlit or Gradio app showing:
  - Reward curve across training stages
  - Side-by-side kernel comparison (base model vs GRPO model)
  - Speedup distribution histograms
  - Live demo: pick an operator → model generates kernel → environment scores it
- Push model + environment to HuggingFace Hub

**Hour 4-5: Pitch**
- 3-minute presentation:
  - "CUDA Agent proved RL trains better kernel writers. DoubleGraph proved per-GPU specialization matters. We combined them at hackathon scale using OpenEnv."
  - Show reward curve going up
  - Show a kernel the model discovered an optimization the SFT data didn't contain
  - "This pipeline is open-source. Plug in any model, any GPU, any operator."

---

## 10. What We Validate

### 10.1 Hypotheses

**H1: Multi-stage RL improves kernel quality over base model.**
Metric: Average reward on held-out evaluation prompts increases from base → post-GRPO.

**H2: RFT warm-up is necessary (replicates CUDA Agent's ablation).**
Metric: GRPO without RFT shows reward collapse; GRPO with RFT shows stable improvement.

**H3: DoubleGraph-informed SKILL.md improves over generic prompts.**
Metric: Kernels generated with A100-specific SKILL.md achieve higher speedup than those with generic "write fast CUDA" prompts.

### 10.2 What "Success" Looks Like

**Minimum viable (must have):**
- OpenEnv environment that compiles, verifies, and benchmarks CUDA kernels for multiple operators
- Multi-stage pipeline runs end-to-end on hackathon hardware
- Evidence that GRPO changes reward distribution (even slightly upward)

**Good:**
- GRPO-trained model consistently generates kernels with reward ≥ 2.0 (beats eager)
- Clear reward curve showing improvement across training stages
- Ablation confirms RFT is necessary (replicates CUDA Agent finding)

**Great:**
- Model discovers optimization patterns through RL that weren't in SFT/RFT data
- Kernels approach or exceed torch.compile performance on some operators
- Framework demonstrated on 3+ different operator types
- Pipeline published as reusable OpenEnv environment on HuggingFace

---

## 11. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| GPTQ model doesn't load on hackathon hardware | Medium | High | Pre-download, have Qwen2.5-Coder-7B as fallback |
| Model generates uncompilable CUDA at high rate | Medium | High | Stage 1 warm-up is specifically designed to fix this |
| GRPO shows no improvement over RFT | Medium | Medium | Valid negative result; present as ablation |
| Training collapses despite RFT | Low | High | CUDA Agent validated this for Seed 1.6; may differ for Qwen3 |
| Hackathon WiFi too slow for 80B model download | High | High | Pre-download to local drive before hackathon |
| Compilation too slow for RL loop | Low | Medium | nvcc is CPU-bound, ~3-5s. Acceptable. |
| GPU is A100 (not H100/B200) | Medium | Medium | Fall back to Qwen2.5-Coder-7B + Unsloth (v4.0 path) |
| TRL GRPOTrainer + GPTQ model incompatibility | Medium | High | Test before hackathon; fall back to Unsloth if needed |

---

## 12. Tech Stack

### 12.1 Dependencies

```
# Training (H100 + GPTQ path)
transformers>=4.48,<5.0
trl>=0.16.0
peft>=0.14.0
gptqmodel>=1.5.0
torch>=2.10.0
datasets>=3.0

# Training (B200 + Unsloth path)
unsloth>=2025.3

# OpenEnv
openenv-core[core]>=0.2.1

# Evaluation
cupy-cuda12x>=14.0
numpy>=1.26

# Data
huggingface-hub>=0.25

# Monitoring
wandb>=0.19
```

### 12.2 Repository Structure

```
KernelForge-OpenEnv/
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── skill.md                              # SKILL.md v5.0 (Section 6.3)
│
├── openenv_env/
│   ├── __init__.py
│   ├── kernel_forge_env.py               # OpenEnv Environment (Section 6.1)
│   ├── cache_pool.py                     # DoubleGraph CachePool
│   ├── reward.py                         # CUDA Agent reward function
│   └── anti_hack.py                      # Symbol scanning, flag whitelist
│
├── training/
│   ├── stage1_warmup.py                  # Single-turn GRPO warm-up
│   ├── stage2_rft.py                     # Trajectory collection + RFT
│   ├── stage3_grpo.py                    # GRPO with curriculum
│   ├── curriculum.py                     # Prompt difficulty management
│   └── model_loader.py                   # GPTQ vs Unsloth loader
│
├── datasets/
│   ├── download_ops6k.py                 # Pull CUDA-Agent-Ops-6K
│   ├── curate_subset.py                  # Select 200-problem subset
│   └── compute_baselines.py             # Pre-compute eager/compile times
│
├── evaluation/
│   ├── ablation.py                       # H1/H2/H3 ablations
│   ├── eval_model.py                     # Evaluate trained model
│   └── compare_stages.py                # Stage-over-stage comparison
│
└── demo/
    └── app.py                            # Streamlit/Gradio demo
```

---

## 13. Corrections from All Prior Versions

| Prior Error | v5.0 Fix |
|-------------|----------|
| "NO SFT JUST RL" — pure RL from base model | Multi-stage: warm-up → RFT → GRPO (CUDA Agent proved pure RL collapses at step 17) |
| DoubleGraph as stretch/reference only | DoubleGraph patterns productized into curriculum, SKILL.md, CachePool, .cu.flags |
| Model: Qwen2.5-Coder-7B (conservative) | Qwen3-Coder-Next (80B MoE, 3B active) with GPTQ or bf16 |
| Hardware: A100 for training | H100/B200 for training, cross-compile for A100 sm_80 |
| WCC-only scope | Multi-algorithm: WCC + operator fusion + general CUDA ops (CUDA-Agent-Ops-6K) |
| Simple 4-tier reward | CUDA Agent's validated {-1, 1, 2, 3} with eager + compile baselines |
| No anti-hacking | Full anti-hacking suite from CUDA Agent Section 3.2 |
| No curriculum | DoubleGraph-inspired 4-phase progression (base → fusion → arch-specific → advanced) |
| FastLanguageModel (dense loader) | AutoModelForCausalLM + PEFT (GPTQ) or FastModel (Unsloth bf16) |
| SFT from Claude/GPT-4 generated data | RFT from model's own trajectories (self-play, as CUDA Agent validated) |

---

## 14. Works Cited

1. Weinan Dai, Hanlin Wu, et al. "CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation." arXiv:2602.24286, February 2026.
2. DoubleAI. "WarpSpeed: Surpassing Expert-Written Kernels At Scale." doubleai.com, March 2, 2026. Architecture analysis: SKILLS.md internal document.
3. BytedTsinghua-SIA. "CUDA-Agent-Ops-6K." HuggingFace Datasets. Apache 2.0.
4. ByteDance-Seed. "cudaLLM-8B." HuggingFace Models. Apache 2.0.
5. NVIDIA. "A100 Tensor Core GPU Architecture Whitepaper." 2020.
6. NVIDIA. "CUDA C++ Programming Guide." CUDA 12.x Documentation.
7. Amnon Shashua, Shai Shalev-Shwartz. "Artificial Expert Intelligence through PAC-reasoning." arXiv:2412.02441, December 2024.
